"""
A/B compare size strategies for classroom grouping cohesion.

Methods
-------
A. posthoc_repair (current production):
   unconstrained K-Medoids (PAM) → medoid-targeted enforce_group_size

B. size_constrained:
   capacity-aware K-Medoids (sizes enforced during assignment + PAM swaps)

Research question: given teacher target size, which yields higher within-group
similarity (lower PAM cost / within L1, higher Manhattan silhouette)?

Usage (repo root, venv active)::

  python -m backend.compare_size_strategies
  python -m backend.compare_size_strategies --csv backend/data/templates/classroom_template.csv --group-size 5
  python -m backend.compare_size_strategies --n 17 --group-size 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from .clustering import (
    compute_feature_vector,
    compute_psychometric_distance_matrix,
    enforce_group_size,
)
from .data_loader import prepare_for_grouping
from .group_config import calculate_n_groups
from .invariants import assert_assignment_invariants
from .kmedoids import (
    capacities_for_target_size,
    kmedoids_pam,
    kmedoids_size_constrained,
)
from .paths import DEFAULT_TEMPLATE_CSV
from .skill_variance import compute_skill_variance


def _recompute_medoids(D: np.ndarray, labels: np.ndarray) -> np.ndarray:
    medoids = []
    for g in sorted(np.unique(labels)):
        idx = np.where(labels == g)[0]
        sub = D[np.ix_(idx, idx)]
        medoids.append(int(idx[int(np.argmin(sub.sum(axis=1)))]))
    return np.asarray(medoids, dtype=int)


def _score(
    name: str,
    X: np.ndarray,
    D: np.ndarray,
    labels: np.ndarray,
    df: pd.DataFrame,
) -> dict[str, Any]:
    medoids = _recompute_medoids(D, labels)
    n = len(labels)
    pam_cost = float(D[np.arange(n), medoids[labels]].sum())

    within = []
    for g in np.unique(labels):
        idx = np.where(labels == g)[0]
        if len(idx) < 2:
            within.append(0.0)
            continue
        sub = D[np.ix_(idx, idx)]
        within.append(float(sub[np.triu_indices(len(idx), k=1)].mean()))

    sizes = [int(np.sum(labels == g)) for g in sorted(np.unique(labels))]
    skill = compute_skill_variance(df, labels)

    return {
        "method": name,
        "sizes": sizes,
        "silhouette_manhattan": float(
            silhouette_score(D, labels, metric="precomputed")
        ),
        "silhouette_euclidean": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "pam_cost": pam_cost,
        "mean_within_pairwise_l1": float(np.mean(within)),
        "skill_variance_mean": float(skill["overall_mean_variance"]),
        "medoid_indices": [int(i) for i in medoids],
    }


def run_posthoc(D: np.ndarray, target_size: int, random_state: int) -> np.ndarray:
    k = calculate_n_groups(len(D), target_size)
    labels, _ = kmedoids_pam(D, k, random_state=random_state)
    return enforce_group_size(labels, target_size, distance_matrix=D)


def run_size_constrained(
    D: np.ndarray, target_size: int, random_state: int
) -> np.ndarray:
    caps = capacities_for_target_size(len(D), target_size)
    labels, _ = kmedoids_size_constrained(D, caps, random_state=random_state)
    return labels


def compare_dataframe(
    df: pd.DataFrame,
    target_size: int,
    *,
    random_state: int = 42,
    label: str = "",
) -> dict[str, Any]:
    X = compute_feature_vector(df)
    D = compute_psychometric_distance_matrix(X)
    n = len(df)
    caps = capacities_for_target_size(n, target_size)

    labels_a = run_posthoc(D, target_size, random_state)
    labels_b = run_size_constrained(D, target_size, random_state)

    assert_assignment_invariants(labels_a, n, target_size)
    assert_assignment_invariants(labels_b, n, target_size)

    score_a = _score("posthoc_repair", X, D, labels_a, df)
    score_b = _score("size_constrained", X, D, labels_b, df)

    # Lower PAM / within L1 / DB / skill_var is better; higher silhouette / CH better.
    cohesion_keys_lower = (
        "pam_cost",
        "mean_within_pairwise_l1",
        "davies_bouldin",
        "skill_variance_mean",
    )
    cohesion_keys_higher = ("silhouette_manhattan", "calinski_harabasz")

    wins = {"posthoc_repair": 0, "size_constrained": 0, "tie": 0}
    deltas: dict[str, float] = {}
    for key in cohesion_keys_lower:
        da = score_b[key] - score_a[key]
        deltas[key] = float(da)
        if abs(da) < 1e-9:
            wins["tie"] += 1
        elif da < 0:
            wins["size_constrained"] += 1
        else:
            wins["posthoc_repair"] += 1
    for key in cohesion_keys_higher:
        da = score_b[key] - score_a[key]
        deltas[key] = float(da)
        if abs(da) < 1e-9:
            wins["tie"] += 1
        elif da > 0:
            wins["size_constrained"] += 1
        else:
            wins["posthoc_repair"] += 1

    # Primary research criterion: within-cluster cohesion under size limits.
    if score_b["pam_cost"] < score_a["pam_cost"] - 1e-9:
        winner = "size_constrained"
    elif score_a["pam_cost"] < score_b["pam_cost"] - 1e-9:
        winner = "posthoc_repair"
    elif score_b["silhouette_manhattan"] > score_a["silhouette_manhattan"] + 1e-9:
        winner = "size_constrained"
    elif score_a["silhouette_manhattan"] > score_b["silhouette_manhattan"] + 1e-9:
        winner = "posthoc_repair"
    else:
        winner = "tie"

    return {
        "dataset": label,
        "n_students": n,
        "target_size": target_size,
        "n_groups": calculate_n_groups(n, target_size),
        "capacities": caps,
        "random_state": random_state,
        "posthoc_repair": score_a,
        "size_constrained": score_b,
        "deltas_constrained_minus_posthoc": deltas,
        "metric_wins": wins,
        "winner_by_cohesion": winner,
        "same_partition": bool(np.array_equal(labels_a, labels_b)),
    }


def _print_case(result: dict[str, Any]) -> None:
    a = result["posthoc_repair"]
    b = result["size_constrained"]
    print("=" * 72)
    print(f"DATASET: {result['dataset']}")
    print(
        f"N={result['n_students']} | target={result['target_size']} | "
        f"k={result['n_groups']} | capacities={result['capacities']}"
    )
    print(f"Same partition? {result['same_partition']}")
    print(f"Winner (cohesion / PAM cost): {result['winner_by_cohesion']}")
    print()
    headers = ("metric", "posthoc_repair", "size_constrained", "delta (B-A)")
    rows = [
        (
            "sizes",
            str(a["sizes"]),
            str(b["sizes"]),
            "",
        ),
        (
            "PAM cost (↓ better)",
            f"{a['pam_cost']:.4f}",
            f"{b['pam_cost']:.4f}",
            f"{result['deltas_constrained_minus_posthoc']['pam_cost']:+.4f}",
        ),
        (
            "Silhouette Man (↑)",
            f"{a['silhouette_manhattan']:.4f}",
            f"{b['silhouette_manhattan']:.4f}",
            f"{result['deltas_constrained_minus_posthoc']['silhouette_manhattan']:+.4f}",
        ),
        (
            "Within pairwise L1 (↓)",
            f"{a['mean_within_pairwise_l1']:.4f}",
            f"{b['mean_within_pairwise_l1']:.4f}",
            f"{result['deltas_constrained_minus_posthoc']['mean_within_pairwise_l1']:+.4f}",
        ),
        (
            "Skill variance (↓)",
            f"{a['skill_variance_mean']:.4f}",
            f"{b['skill_variance_mean']:.4f}",
            f"{result['deltas_constrained_minus_posthoc']['skill_variance_mean']:+.4f}",
        ),
        (
            "Davies-Bouldin (↓)",
            f"{a['davies_bouldin']:.4f}",
            f"{b['davies_bouldin']:.4f}",
            f"{result['deltas_constrained_minus_posthoc']['davies_bouldin']:+.4f}",
        ),
        (
            "Calinski-Harabasz (↑)",
            f"{a['calinski_harabasz']:.4f}",
            f"{b['calinski_harabasz']:.4f}",
            f"{result['deltas_constrained_minus_posthoc']['calinski_harabasz']:+.4f}",
        ),
    ]
    print(f"{headers[0]:26s} {headers[1]:>16s} {headers[2]:>18s} {headers[3]:>14s}")
    print("-" * 72)
    for row in rows:
        print(f"{row[0]:26s} {row[1]:>16s} {row[2]:>18s} {row[3]:>14s}")
    print("=" * 72)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare post-hoc size repair vs size-constrained K-Medoids"
    )
    parser.add_argument("--csv", default=str(DEFAULT_TEMPLATE_CSV))
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n",
        type=int,
        default=0,
        help="If >0, use only the first N rows of the CSV (e.g. 17 for remainder case).",
    )
    parser.add_argument(
        "--also-low-scores",
        action="store_true",
        help="Also run low_scores_template.csv at the same group size.",
    )
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []

    df = prepare_for_grouping(args.csv)
    if args.n and args.n < len(df):
        df = df.iloc[: args.n].reset_index(drop=True)
        label = f"{Path(args.csv).stem}_n{args.n}"
    else:
        label = Path(args.csv).stem

    result = compare_dataframe(
        df, args.group_size, random_state=args.seed, label=label
    )
    _print_case(result)
    cases.append(result)

    # Always include the classic remainder example when not already that case.
    if not (args.n == 17 and args.group_size == 5):
        df17 = prepare_for_grouping(str(DEFAULT_TEMPLATE_CSV)).iloc[:17].reset_index(
            drop=True
        )
        r17 = compare_dataframe(
            df17, 5, random_state=args.seed, label="classroom_template_n17_target5"
        )
        _print_case(r17)
        cases.append(r17)

    if args.also_low_scores:
        low = prepare_for_grouping(
            "backend/data/templates/low_scores_template.csv"
        )
        rlow = compare_dataframe(
            low, args.group_size, random_state=args.seed, label="low_scores_template"
        )
        _print_case(rlow)
        cases.append(rlow)

    constrained_wins = sum(
        1 for c in cases if c["winner_by_cohesion"] == "size_constrained"
    )
    posthoc_wins = sum(1 for c in cases if c["winner_by_cohesion"] == "posthoc_repair")
    ties = sum(1 for c in cases if c["winner_by_cohesion"] == "tie")
    print("SUMMARY")
    print(
        f"  size_constrained wins: {constrained_wins} | "
        f"posthoc_repair wins: {posthoc_wins} | ties: {ties}"
    )
    print(
        "  Primary criterion: lower PAM cost (total Manhattan distance to medoids), "
        "then higher Manhattan silhouette."
    )

    out_dir = Path("backend/output/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        Path(args.json_out)
        if args.json_out
        else out_dir / "size_strategy_comparison.json"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"cases": cases, "summary": {
            "size_constrained_wins": constrained_wins,
            "posthoc_repair_wins": posthoc_wins,
            "ties": ties,
        }}, f, indent=2)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
