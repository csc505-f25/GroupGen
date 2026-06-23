"""
CLI group generator — same pipeline as the study API, with disk output.

Writes per-run artifacts under ``backend/output/runs/<timestamp>_<uuid>/``:
  - ``final_groups.csv`` — roster with Group_ID
  - ``final_groups_report.txt`` — printable summary
  - ``run_manifest.json`` — audit metadata (see ``run_manifest.py``)

Environment overrides: ``GROUPGEN_INPUT_CSV``, ``GROUPGEN_OUTPUT_CSV``.
"""

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import DEFAULT_CLI_OUTPUT_DIR, DEFAULT_TEMPLATE_CSV
from .data_loader import prepare_for_grouping, read_csv_source
from .group_gen_intake import is_raw_google_form, merge_groups_into_raw_export
from .group_config import calculate_n_groups
from .invariants import InvariantViolation
from .pipeline import DEFAULT_RANDOM_STATE, run_grouping_pipeline
from .run_manifest import write_run_manifest

DEFAULT_INPUT = DEFAULT_TEMPLATE_CSV
DEFAULT_OUTPUT_DIR = DEFAULT_CLI_OUTPUT_DIR


def make_run_output_paths(base_dir: Path = DEFAULT_OUTPUT_DIR) -> tuple[Path, Path, Path]:
    """Unique run folder with CSV + report paths so concurrent runs do not overwrite."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_id = f"{stamp}_{uuid.uuid4().hex[:8]}"  # study audit: one folder per execution
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "final_groups.csv"
    report_path = run_dir / "final_groups_report.txt"
    return run_dir, csv_path, report_path


def get_user_target_size() -> int:
    while True:
        try:
            user_input = input("\nEnter target number of students per group (e.g., 5): ")
            size = int(user_input)
            if size <= 0:
                print("Error: Please enter a number greater than 0.")
                continue
            return size
        except ValueError:
            print("Invalid input. Please enter a whole number.")


def save_results(
    df: pd.DataFrame,
    labels: np.ndarray,
    output_file: Path,
    *,
    raw_source: pd.DataFrame | None = None,
) -> None:
    # Persist human-readable Group_ID (1-based) alongside original survey columns.
    df_final = df.copy()
    df_final["Group_ID"] = np.array(labels).astype(int) + 1
    cols = ["Group_ID"] + [c for c in df_final.columns if c != "Group_ID"]
    df_final = df_final[cols].sort_values(by=["Group_ID", "Name"])

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"\nSUCCESS! Groups saved to: {output_path}")

    print("   > Generating text report from saved CSV...")
    df_clean = pd.read_csv(output_path)
    report_path = output_path.with_name(output_path.stem + "_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        def write_line(text: str = "") -> None:
            print(text)
            f.write(text + "\n")

        write_line("\n" + "=" * 65)
        write_line("FINAL GROUP ASSIGNMENTS REPORT")
        write_line("=" * 65)

        for g_id in sorted(df_clean["Group_ID"].unique()):
            group_data = df_clean[df_clean["Group_ID"] == g_id]
            write_line(f"\n GROUP {g_id} ({len(group_data)} students)")
            write_line("-" * 65)
            write_line(f"{'Name':<20} | {'Gender':<8} | {'Ethnicity':<20} | {'Style':<10}")
            write_line("-" * 65)
            for _, row in group_data.iterrows():
                diversity = str(row["Diversity"])
                if len(diversity) > 19:
                    diversity = diversity[:17] + ".."
                write_line(
                    f"{str(row['Name']):<20} | {str(row['Gender']):<8} | "
                    f"{diversity:<20} | {str(row['Learning_Style']):<10}"
                )
            write_line("-" * 65)

    print(f"Readable report saved to: {report_path}")

    if raw_source is not None:
        raw_path = output_path.with_name("raw_responses_with_groups.csv")
        raw_with_groups = merge_groups_into_raw_export(raw_source, df_clean)
        raw_with_groups.to_csv(raw_path, index=False)
        print(f"Original survey export + Group_ID saved to: {raw_path}")


def main() -> None:
    print("\n" + "=" * 60)
    print("GROUPGEN — FINAL GROUP GENERATION")
    print("=" * 60)

    input_path = os.environ.get("GROUPGEN_INPUT_CSV", str(DEFAULT_INPUT))
    run_dir: Path

    if os.environ.get("GROUPGEN_OUTPUT_CSV"):
        # Custom path still gets manifest in the same directory as the CSV.
        output_csv = Path(os.environ["GROUPGEN_OUTPUT_CSV"])
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        run_dir = output_csv.parent
    else:
        run_dir, output_csv, _ = make_run_output_paths()

    print(f"1. Loading data from: {input_path}")
    raw_df = read_csv_source(input_path)
    raw_for_merge = raw_df if is_raw_google_form(raw_df) else None
    df = prepare_for_grouping(raw_df)  # same ingest as API (includes Google Form transform)

    target_size = get_user_target_size()
    n_groups = calculate_n_groups(len(df), target_size)
    print(f"   > Will form {n_groups} groups for {len(df)} students.")

    try:
        result = run_grouping_pipeline(df, target_size, verbose=True)
    except InvariantViolation as e:
        print(f"\nERROR: Group assignment failed validation: {e}")
        raise SystemExit(1) from e
    except ValueError as e:
        print(f"\nERROR: {e}")
        raise SystemExit(1) from e

    if result.warnings:
        print("\n--- Warnings (review these groups manually) ---")
        for w in result.warnings:
            print(f"  • {w}")

    print(f"   > Group sizes: {result.group_size_range}")

    save_results(df, result.labels, output_csv, raw_source=raw_for_merge)

    manifest = write_run_manifest(
        run_dir,
        input_path=str(input_path),
        n_students=len(df),
        target_group_size=target_size,
        n_groups=n_groups,
        random_state=DEFAULT_RANDOM_STATE,
        warnings=result.warnings,
        output_csv=str(output_csv),
    )
    print(f"   > Run manifest: {manifest}")


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit(
            "Run from the repo root:  python -m backend.generate_groups"
        )
    main()
