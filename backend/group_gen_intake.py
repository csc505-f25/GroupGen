import pandas as pd
import numpy as np
import sys
import os

# --- Scoring helpers ---------------------------------------------------------

def _mean_to_1_4_mot_we(mean_val: float) -> int:
    """
    NEW METHOD: Maps a continuous mean (1-4) into 4 discrete buckets.
    This solves the issue of converting a 3-tier scoring manual into a 1-4 scale.
    Because it uses the mean, it works perfectly whether a section has 9 questions 
    or 17 questions, preventing "sum inflation".
    """
    if mean_val <= 2.0:
        return 1
    elif mean_val <= 2.6:
        return 2
    elif mean_val <= 3.2:
        return 3
    else:
        return 4


def _mean_to_1_4_se(mean_val: float) -> int:
    """Maps a 1-7 mean score to 1-4 scale (used for Self-Esteem)."""
    if mean_val <= 2.5:
        return 1
    elif mean_val <= 4.0:
        return 2
    elif mean_val <= 5.5:
        return 3
    else:
        return 4


def _learning_style(row: pd.Series) -> str:
    """Counts A/B/C answers and returns the majority learning style."""
    vals = row.values
    visual      = np.sum(vals == 'A')
    auditory    = np.sum(vals == 'B')
    kinesthetic = np.sum(vals == 'C')
    counts = {'Visual': visual, 'Auditory': auditory, 'Kinesthetic': kinesthetic}
    return max(counts, key=counts.get)


# --- Main processor ----------------------------------------------------------

def process_google_form(input_source, output_path: str = None) -> pd.DataFrame:
    """
    Reads a raw Google Form CSV export and converts it to GroupGen format.
    Dynamically finds column ranges based on text markers.
    """
    # -- Load ----------------------------------------------------------------
    df = pd.read_csv(input_source, dtype=str)
    headers = df.columns.tolist()

    # --- DYNAMIC INDEX FINDING ---
    def find_index(keyword, exact=False):
        """Helper to find column index by searching header text."""
        for i, h in enumerate(headers):
            if exact:
                if keyword.lower() == h.lower().strip(): return i
            else:
                if keyword.lower() in h.lower(): return i
        return -1

    # Find the "Magic Wand" dividers
    wand_indices = [i for i, h in enumerate(headers) if "magic wand" in h.lower()]
    
    # Identify Core Demographic Columns
    name_idx = find_index("first and last name")
    if name_idx == -1: name_idx = 1 # Fallback to standard col 1
    
    email_idx = find_index("email")
    gender_idx = find_index("gender identity")
    diversity_idx = find_index("ethnicity")

    # -------------------------------------------------------------------------
    # Identify Trait Sections (Immune to inserted/deleted columns)
    # -------------------------------------------------------------------------
    
    # 1. Learning Style (Starts after Email, ends at the first Magic Wand)
    ls_start = email_idx + 1 if email_idx != -1 else 3
    ls_end = wand_indices[0] if len(wand_indices) > 0 else 31

    # 2. Self-Esteem (Starts after first Magic Wand)
    se_start = wand_indices[0] + 1
    # Ends at the start of the Motivation section
    se_end = find_index("not the type to do well in computer programming")
    
    # 3. Motivation (Starts at 'not the type', ends at the NEXT Magic Wand)
    mot_start = se_end
    # Find the wand that comes immediately after Motivation
    mot_end = next((i for i in wand_indices if i > mot_start), mot_start + 17)
    
    # 4. Work Ethic (Starts at 'arrive at classes')
    # Note: Skipped the extra questions (59-65) to strictly grab Work Ethic
    we_start = find_index("arrive at classes and other meetings on time")
    # Find the wand that comes immediately after Work Ethic
    we_end = next((i for i in wand_indices if i > we_start), we_start + 9)


    # -- Clean Data ----------------------------------------------------------
    # Drop rows where name is empty
    valid_rows_mask = df.iloc[:, name_idx].fillna("").astype(str).str.strip() != ""
    df = df[valid_rows_mask].copy()

    n = len(df)
    source_name = input_source if isinstance(input_source, str) else "uploaded file"
    print(f"Loaded {n} student responses from: {source_name}")

    out = pd.DataFrame()
    out['Name'] = df.iloc[:, name_idx].fillna("Unknown").astype(str).str.strip()

    # -- Learning Style ------------------------------------------------------
    ls_cols = df.iloc[:, ls_start:ls_end].copy()
    
    def _extract_abc(val):
        if pd.isna(val): return "UNKNOWN"
        val_clean = str(val).strip().upper()
        if val_clean.startswith("A"): return "A"
        if val_clean.startswith("B"): return "B"
        if val_clean.startswith("C"): return "C"
        return "UNKNOWN"

    ls_cols = ls_cols.applymap(_extract_abc)
    out['Learning_Style'] = ls_cols.apply(_learning_style, axis=1)

    # -- Self-Esteem (1-7 mean -> 1-4) ---------------------------------------
    se_cols = df.iloc[:, se_start:se_end].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    out['Self_Esteem'] = se_cols.mean(axis=1).round(2).apply(_mean_to_1_4_se)

    # -- Motivation (Mean -> 1-4 scale) --------------------------------------
    mot_cols = df.iloc[:, mot_start:mot_end].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    out['Motivation'] = mot_cols.mean(axis=1).round(2).apply(_mean_to_1_4_mot_we)

    # -- Work Ethic (Mean -> 1-4 scale) --------------------------------------
    we_cols = df.iloc[:, we_start:we_end].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    out['Work_Ethic'] = we_cols.mean(axis=1).round(2).apply(_mean_to_1_4_mot_we)

    # -- Demographics --------------------------------------------------------
    out['Gender'] = df.iloc[:, gender_idx].str.strip() if gender_idx != -1 else "Unknown"
    out['Diversity'] = df.iloc[:, diversity_idx].str.strip() if diversity_idx != -1 else "Unknown"

    # -- Reorder columns (Original Format) -----------------------------------
    out = out[[
        'Name', 'Gender', 'Diversity', 'Learning_Style',
        'Motivation', 'Self_Esteem', 'Work_Ethic'
    ]]

    # -- Save ----------------------------------------------------------------
    if output_path is None and isinstance(input_source, str):
        base_dir = os.path.dirname(os.path.abspath(input_source))
        output_path = os.path.join(base_dir, 'groupgen_output.csv')

    if output_path:
        out.to_csv(output_path, index=False)
        print(f"GroupGen CSV saved to: {output_path}")

    return out