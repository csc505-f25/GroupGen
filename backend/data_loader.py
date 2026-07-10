"""
CSV ingestion and validation for GroupGen.

**Production entry point:** ``prepare_for_grouping`` — call this before
``run_grouping_pipeline`` from the API, CLI, or evaluation scripts.

Pipeline inside ``prepare_for_grouping``:
  read → (raw Google Form? → ``group_gen_intake.process_google_form``)
  → normalize_column_names → drop_empty_rows → validate_data
  → preprocess_data → validate_data (again)

``load_student_data`` only checks columns exist; it does not run full validation.
Use ``prepare_for_grouping`` for study-grade ingest.
"""

import csv
import io
from pathlib import Path
from typing import List, Union

import pandas as pd

VALID_LEARNING_STYLES = {"Visual", "Auditory", "Kinesthetic"}
REQUIRED_COLUMNS = [
    "Name",
    "Gender",
    "Motivation",
    "Self_Esteem",
    "Work_Ethic",
    "Learning_Style",
    "Diversity",
]

# Common Google Forms / spreadsheet header variants -> canonical names
COLUMN_ALIASES = {
    "name": "Name",
    "student name": "Name",
    "full name": "Name",
    "your name": "Name",
    "gender": "Gender",
    "sex": "Gender",
    "motivation": "Motivation",
    "self esteem": "Self_Esteem",
    "self_esteem": "Self_Esteem",
    "self-esteem": "Self_Esteem",
    "work ethic": "Work_Ethic",
    "work_ethic": "Work_Ethic",
    "work-ethic": "Work_Ethic",
    "learning style": "Learning_Style",
    "learning_style": "Learning_Style",
    "diversity": "Diversity",
    "ethnicity": "Diversity",
    "race": "Diversity",
    "race/ethnicity": "Diversity",
}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip headers and map known aliases (Google Forms friendly)."""
    df = df.copy()
    cleaned = []
    for col in df.columns:
        name = str(col).strip().lstrip("\ufeff")  # UTF-8 BOM on first column
        key = name.lower()
        cleaned.append(COLUMN_ALIASES.get(key, name))
    df.columns = cleaned
    return df


def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove blank rows often trailing Google Forms exports."""
    if "Name" not in df.columns:
        return df.dropna(how="all")
    mask = df["Name"].astype(str).str.strip().replace("", pd.NA).notna()
    return df.loc[mask].reset_index(drop=True)


def _merge_adjacent_pair(row: List[str], pos: int) -> List[str]:
    """Join row[pos] and row[pos+1] with a comma."""
    return row[:pos] + [f"{row[pos]},{row[pos + 1]}"] + row[pos + 2 :]


def _repair_vak_splits(
    row: List[str],
    ncols: int,
    merge_start: int,
    merge_end: int,
) -> List[str]:
    """
    Rejoin cells split by commas inside one VAK answer (e.g. option 22c with
    \"such as an activity or a meal\").
    """
    from .vak_answer_catalog import is_known_vak_answer

    row = list(row)
    # When the row is wider than the header, scan all spill cells (not just header VAK cols)
    scan_end = len(row) if len(row) > ncols else min(merge_end, len(row))
    while len(row) > ncols and merge_start < scan_end:
        fixed = False
        for pos in range(merge_start, scan_end):
            for width in range(2, min(9, merge_end - pos + 2)):
                if pos + width > len(row):
                    break
                chunk = ",".join(row[pos : pos + width])
                if is_known_vak_answer(chunk):
                    row = row[:pos] + [chunk] + row[pos + width :]
                    fixed = True
                    break
            if fixed:
                break
        if not fixed:
            break
    return row


def _read_csv_rows(handle) -> List[List[str]]:
    """Parse CSV with the stdlib (RFC 4180 quoting); repair ragged rows."""
    reader = csv.reader(handle, delimiter=",", quotechar='"', doublequote=True)
    rows = list(reader)
    if not rows:
        return rows
    header = rows[0]
    ncols = len(header)
    merge_start = 3
    merge_end = ncols
    for i, h in enumerate(header):
        hlow = str(h).lower()
        if "first and last name" in hlow:
            merge_start = i + 1
        if "magic wand" in hlow and merge_end == ncols:
            merge_end = i
    fixed = [header]
    for line_no, row in enumerate(rows[1:], start=2):
        if len(row) != ncols:
            # Prefer catalog joins (split VAK answers); then pairwise merges in VAK band
            for _ in range(len(row)):
                prev_len = len(row)
                row = _repair_vak_splits(row, ncols, merge_start, merge_end)
                if len(row) == prev_len and len(row) > ncols:
                    # Drop stray empty fields (e.g. blank magic-wand cells)
                    trimmed = [c for c in row if str(c).strip() != ""]
                    if len(trimmed) < len(row):
                        row = trimmed
                    else:
                        break
                if len(row) == ncols:
                    break
        if len(row) < ncols:
            row = row + [""] * (ncols - len(row))
        row = row[:ncols]
        if len(row) != ncols:
            raise ValueError(
                f"CSV row {line_no} has {len(row)} columns but the header has {ncols}. "
                "Many VAK answers contain commas (e.g. \"try to get together whilst "
                "doing something else, such as an activity or a meal\"). "
                "Re-export from Google Forms (Responses → Download CSV) without "
                "opening the file in Excel first."
            )
        fixed.append(row)
    return fixed


def _read_csv_repaired(text: str) -> pd.DataFrame:
    rows = _read_csv_rows(io.StringIO(text))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows[1:], columns=rows[0])


def read_csv_source(source: Union[str, Path, bytes]) -> pd.DataFrame:
    """Read CSV from file path or raw bytes (API upload).

    Tries pandas first (types + quoting). On ragged rows (e.g. \"Expected 49
    fields, saw 63\"), re-parses with repair for commas inside answer text.
    """
    if isinstance(source, (str, Path)):
        raw = Path(source).read_text(encoding="utf-8-sig")
    else:
        raw = source.decode("utf-8-sig")

    if not raw.strip():
        return pd.DataFrame()

    # Google Form exports: many VAK answers contain commas; pandas often mis-aligns
    # columns while still parsing. Prefer the repair pass that rejoins catalog text.
    if "first and last name" in raw.lower():
        return _read_csv_repaired(raw)

    buf = io.StringIO(raw)
    for kwargs in (
        {"sep": ",", "engine": "python"},
        {"sep": None, "engine": "python"},
    ):
        try:
            buf.seek(0)
            return pd.read_csv(buf, encoding="utf-8-sig", **kwargs)
        except pd.errors.ParserError:
            continue

    return _read_csv_repaired(raw)


def load_student_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load student data from a CSV file path."""
    try:
        df = read_csv_source(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error loading CSV: {e}") from e

    df = normalize_column_names(df)
    df = drop_empty_rows(df)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Required: {REQUIRED_COLUMNS}. "
            "Tip: use backend/data/templates/classroom_template.csv as a guide."
        )
    return df


def prepare_for_grouping(source: Union[str, Path, bytes, pd.DataFrame]) -> pd.DataFrame:
    """
    Full ingest path for the study: read -> normalize columns -> drop blanks ->
    validate -> preprocess.

    Raises ValueError with teacher-friendly messages on failure.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = read_csv_source(source)

    # Raw Google Forms export → seven clustering columns (teachers upload as-is).
    from .group_gen_intake import is_raw_google_form, process_google_form

    if is_raw_google_form(df):
        try:
            df = process_google_form(df)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Could not parse Google Form export: {e}") from e

    # --- Structural cleanup (no row drops based on scores yet) ---
    df = normalize_column_names(df)
    df = drop_empty_rows(df)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        found = [str(c) for c in df.columns.tolist()]
        hint = ""
        if len(found) == 1 and "," in found[0]:
            hint = " Your file looks like one column — re-export as CSV (comma-separated), not Excel (.xlsx)."
        elif len(found) == 1:
            hint = (
                f" Your file parsed as a single column ({found[0][:60]}...). "
                "Re-export from Google Forms (Responses → Download CSV) or save Excel as "
                "CSV UTF-8 (comma delimited)."
            )
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Your CSV must include: {', '.join(REQUIRED_COLUMNS)}. "
            f"Columns found in your file: {found}. "
            "Extra columns (e.g. Timestamp from Google Forms) are fine."
            + hint
        )

    # First pass: catch obvious problems on raw/normalized strings.
    validation = validate_data(df)
    if not validation["is_valid"]:
        raise ValueError("; ".join(validation["errors"]))

    df = preprocess_data(df, log=False)

    # Second pass: catch NaNs introduced by coercion (e.g. "N/A" in Motivation).
    validation = validate_data(df)
    if not validation["is_valid"]:
        raise ValueError(
            "Data could not be cleaned automatically: " + "; ".join(validation["errors"])
        )

    return df


def validate_data(df: pd.DataFrame) -> dict:
    """Validate required columns and value ranges."""
    errors: List[str] = []
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")

    for col in ["Motivation", "Self_Esteem", "Work_Ethic"]:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"{col} must be numeric (use 1–4).")
        elif not df[col].between(1, 4).all():
            errors.append(f"{col} must only contain values from 1 to 4.")

    if df.isna().to_numpy().any():
        errors.append("Data contains empty cells; fill or remove those rows.")

    if "Name" in df.columns:
        dupes = df["Name"].astype(str).duplicated()
        if dupes.any():
            errors.append("Duplicate student names found; each Name must be unique.")

    if "Learning_Style" in df.columns:
        styles = df["Learning_Style"].astype(str).str.strip().str.title()
        invalid = styles[~styles.isin(VALID_LEARNING_STYLES)]
        if not invalid.empty:
            errors.append(
                "Learning_Style must be Visual, Auditory, or Kinesthetic. "
                f"Invalid: {invalid.unique().tolist()}"
            )

    if "Gender" in df.columns:
        genders = df["Gender"].astype(str).str.strip()
        empty_gender = genders.replace("", pd.NA).isna()
        if empty_gender.any():
            errors.append("Gender cannot be empty.")

    if "Diversity" in df.columns:
        diversity = df["Diversity"].astype(str).str.strip()
        empty_div = diversity.replace("", pd.NA).isna()
        if empty_div.any():
            errors.append(
                "Diversity cannot be empty; use a consistent ethnicity/category label."
            )

    return {"is_valid": len(errors) == 0, "errors": errors}


def preprocess_data(df: pd.DataFrame, *, log: bool = True) -> pd.DataFrame:
    """Clean and normalize student data for clustering."""
    missing = df.isnull()
    if log and missing.any().any():
        print("=== MISSING DATA FOUND ===")
        for col in df.columns:
            if missing[col].any():
                print(f"Column '{col}' has missing values at rows: {df.index[missing[col]].tolist()}")
    elif log:
        print("No missing data found.")

    df_cleaned = df.copy()

    df_cleaned["Name"] = df_cleaned["Name"].astype(str).str.strip()
    df_cleaned["Gender"] = (
        df_cleaned["Gender"]
        .astype(str)
        .str.strip()
        .str.replace(r"^(m|male)$", "Male", regex=True, case=False)
        .str.replace(r"^(f|female)$", "Female", regex=True, case=False)
    )
    # Title-case other values (e.g. Non-binary) without breaking Male/Female
    other = ~df_cleaned["Gender"].isin(["Male", "Female"])
    df_cleaned.loc[other, "Gender"] = df_cleaned.loc[other, "Gender"].str.title()

    df_cleaned["Learning_Style"] = (
        df_cleaned["Learning_Style"].astype(str).str.strip().str.title()
    )
    df_cleaned["Diversity"] = df_cleaned["Diversity"].astype(str).str.strip().str.title()

    # Coerce scores to numbers; invalid text becomes NaN and fails second validate_data.
    num_cols = ["Motivation", "Self_Esteem", "Work_Ethic"]
    df_cleaned[num_cols] = df_cleaned[num_cols].apply(pd.to_numeric, errors="coerce")

    return df_cleaned
