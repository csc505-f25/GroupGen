"""Batch A/B size-strategy comparison across all available input CSVs."""

from __future__ import annotations

import json
from pathlib import Path

from backend.compare_size_strategies import _print_case, compare_dataframe
from backend.data_loader import prepare_for_grouping
from backend.group_config import calculate_n_groups

TARGET = 5
SEED = 42


def main() -> None:
    root = Path("docs/Inclass_data")
    templates = Path("backend/data/templates")
    candidates = sorted(root.rglob("*.csv")) + [
        templates / "classroom_template.csv",
        templates / "low_scores_template.csv",
        templates / "google_form_sample.csv",
    ]

    cases = []
    skipped = []

    for path in candidates:
        rel = path.as_posix()
        try:
            df = prepare_for_grouping(path)
        except Exception as e:  # noqa: BLE001 - batch resilience
            skipped.append({"file": rel, "reason": f"intake failed: {e}"})
            print(f"SKIP intake: {path.name}: {e}")
            continue

        n = len(df)
        if n < 4:
            skipped.append({"file": rel, "reason": f"too few students ({n})"})
            print(f"SKIP n={n}: {path.name}")
            continue

        k = calculate_n_groups(n, TARGET)
        if k < 2 or k >= n:
            skipped.append({"file": rel, "reason": f"bad k={k} for n={n}"})
            print(f"SKIP k={k}: {path.name}")
            continue

        try:
            result = compare_dataframe(
                df, TARGET, random_state=SEED, label=path.stem
            )
            _print_case(result)
            cases.append(result)
        except Exception as e:  # noqa: BLE001
            skipped.append({"file": rel, "reason": f"compare failed: {e}"})
            print(f"SKIP compare: {path.name}: {e}")

    b_wins = sum(1 for c in cases if c["winner_by_cohesion"] == "size_constrained")
    a_wins = sum(1 for c in cases if c["winner_by_cohesion"] == "posthoc_repair")
    ties = sum(1 for c in cases if c["winner_by_cohesion"] == "tie")

    print("=" * 72)
    print("OVERALL SUMMARY")
    print(f"  Ran: {len(cases)} datasets | Skipped: {len(skipped)}")
    print(f"  size_constrained (B) wins: {b_wins}")
    print(f"  posthoc_repair (A) wins:   {a_wins}")
    print(f"  ties:                      {ties}")
    print()
    header = (
        f"{'dataset':55s} {'N':>3s} {'PAM A':>8s} {'PAM B':>8s} "
        f"{'SilA':>6s} {'SilB':>6s} {'winner':>16s}"
    )
    print(header)
    for c in cases:
        a, b = c["posthoc_repair"], c["size_constrained"]
        print(
            f"{c['dataset'][:55]:55s} {c['n_students']:3d} "
            f"{a['pam_cost']:8.3f} {b['pam_cost']:8.3f} "
            f"{a['silhouette_manhattan']:6.3f} {b['silhouette_manhattan']:6.3f} "
            f"{c['winner_by_cohesion']:>16s}"
        )

    if skipped:
        print()
        print("SKIPPED:")
        for s in skipped:
            print(f"  - {s['file']}: {s['reason']}")

    out = Path("backend/output/diagnostics/size_strategy_all_inputs.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "target_size": TARGET,
                "seed": SEED,
                "cases": cases,
                "skipped": skipped,
                "summary": {"B_wins": b_wins, "A_wins": a_wins, "ties": ties},
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
