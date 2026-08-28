"""
Benchmark CPU vs GPU backends for GroupGen.

Usage (from repo root):
  python -m backend.benchmark_backends
  python -m backend.benchmark_backends --sizes 100 500 1000 --group-size 10
  GROUPGEN_BACKEND=gpu python -m backend.benchmark_backends
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .compute_backend import list_available_backends, resolve_backend
from .data_loader import prepare_for_grouping
from .paths import DEFAULT_GOOGLE_FORM_SAMPLE_CSV
from .pipeline import run_grouping_pipeline


def _load_base_roster(path: Path) -> pd.DataFrame:
    return prepare_for_grouping(path.read_bytes())


def _expand_roster(df: pd.DataFrame, n_students: int) -> pd.DataFrame:
    """Duplicate and lightly vary students to reach ``n_students`` for scaling tests."""
    if n_students <= len(df):
        return df.head(n_students).copy()

    chunks = []
    copies_needed = (n_students + len(df) - 1) // len(df)
    for i in range(copies_needed):
        chunk = df.copy()
        if i > 0:
            chunk["Name"] = chunk["Name"] + f" #{i + 1}"
            # Slight score jitter so repeated templates are not identical rows.
            for col in ("Motivation", "Self_Esteem", "Work_Ethic"):
                if col in chunk.columns:
                    delta = (i % 3) - 1  # -1, 0, or +1
                    chunk[col] = (chunk[col] + delta).clip(1, 4)
        chunks.append(chunk)
    big = pd.concat(chunks, ignore_index=True)
    return big.iloc[:n_students].copy()


def _run_once(df: pd.DataFrame, group_size: int, backend: str) -> dict:
    resolved = resolve_backend(backend)
    t0 = time.perf_counter()
    result = run_grouping_pipeline(df, group_size, verbose=False, backend=backend)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "backend": result.backend,
        "label": result.backend_label,
        "wall_ms": round(wall_ms, 2),
        "timing": result.timing_ms,
        "labels": result.labels.copy(),
        "n_groups": result.n_groups,
    }


def _labels_match(a: np.ndarray, b: np.ndarray) -> bool:
    return np.array_equal(a, b)


def run_benchmark(
    *,
    sizes: list[int],
    group_size: int,
    source_csv: Path,
    backends: list[str],
    repeats: int = 3,
) -> None:
    base = _load_base_roster(source_csv)
    print("GroupGen backend benchmark")
    print(f"Source: {source_csv} ({len(base)} students in template)")
    print(f"Available on this machine: {', '.join(list_available_backends())}")
    print(f"Testing backends: {', '.join(backends)}")
    print(f"Group size target: {group_size}")
    print()

    for n in sizes:
        df = _expand_roster(base, n)
        print(f"=== {n} students ===")
        rows: list[dict] = []
        reference_labels: np.ndarray | None = None

        for backend in backends:
            try:
                samples = []
                last_labels = None
                for _ in range(repeats):
                    out = _run_once(df, group_size, backend)
                    samples.append(out["wall_ms"])
                    last_labels = out["labels"]

                median_ms = float(np.median(samples))
                row = {
                    "backend": out["backend"],
                    "label": out["label"],
                    "median_ms": round(median_ms, 2),
                    "distance_ms": out["timing"].get("distance_matrix"),
                    "kmedoids_ms": out["timing"].get("kmedoids"),
                    "pipeline_ms": out["timing"].get("total_pipeline"),
                }
                rows.append(row)

                if reference_labels is None:
                    reference_labels = last_labels
                elif last_labels is not None and not _labels_match(reference_labels, last_labels):
                    print(
                        f"  NOTE: {backend} labels differ from first backend "
                        "(float/device tie-breaking can cause this at large N)."
                    )
            except RuntimeError as exc:
                print(f"  {backend}: SKIP — {exc}")
                continue

        if not rows:
            print("  (no backends ran)\n")
            continue

        cpu_row = next((r for r in rows if r["backend"] == "cpu"), rows[0])
        for r in rows:
            speedup = cpu_row["median_ms"] / r["median_ms"] if r["median_ms"] > 0 else 0
            verdict = (
                f"{speedup:.2f}x faster than CPU"
                if speedup > 1.05
                else (
                    f"{1 / speedup:.2f}x slower than CPU"
                    if speedup < 0.95
                    else "~same as CPU"
                )
            )
            print(
                f"  {r['label']}: median {r['median_ms']} ms "
                f"(dist {r['distance_ms']} ms, kmedoids {r['kmedoids_ms']} ms) — {verdict}"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GroupGen CPU vs GPU backends.")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 200, 500, 1000],
        help="Roster sizes to test",
    )
    parser.add_argument("--group-size", type=int, default=10, help="Target students per group")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_GOOGLE_FORM_SAMPLE_CSV,
        help="Template CSV to scale up for tests",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["cpu", "torch_cpu", "gpu"],
        help="Backends to compare (cpu, torch_cpu, gpu, cuda, directml)",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Runs per backend for median")
    args = parser.parse_args()

    run_benchmark(
        sizes=args.sizes,
        group_size=args.group_size,
        source_csv=args.csv,
        backends=args.backends,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
