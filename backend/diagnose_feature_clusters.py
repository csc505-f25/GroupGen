"""
Feature-only cluster health diagnostic for production GroupGen.

Scores labels from ``run_grouping_pipeline`` (and raw PAM before size balance)
under the same Manhattan geometry used in production.

Usage (from repo root, with venv active):
  python -m backend.diagnose_feature_clusters
  python -m backend.diagnose_feature_clusters --csv backend/data/templates/low_scores_template.csv --group-size 5
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
    pairwise_distances,
    silhouette_score,
)

from .clustering import (
    compute_feature_vector,
    compute_psychometric_distance_matrix,
    enforce_group_size,
)
from .data_loader import prepare_for_grouping
from .group_config import calculate_n_groups
from .kmedoids import kmedoids_pam
from .paths import DEFAULT_TEMPLATE_CSV
from .pipeline import run_grouping_pipeline
from .skill_variance import compute_skill_variance

FEATURE_NAMES = [
    "Motivation_z",
    "Self_Esteem_z",
    "Work_Ethic_z",
    "LS_Visual",
    "LS_Auditory",
    "LS_Kinesthetic",
]
RAW_SKILL_COLS = ["Motivation", "Self_Esteem", "Work_Ethic"]


def _pam_cost(distance_matrix: np.ndarray, labels: np.ndarray, medoid_indices: np.ndarray) -> float:
    """Sum of distances to assigned medoids (PAM objective / L1 'inertia')."""
    n = len(labels)
    medoids_for_points = medoid_indices[labels]
    return float(distance_matrix[np.arange(n), medoids_for_points].sum())


def _nearest_medoids(distance_matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Recompute geometric medoids of current assignments (argmin within-cluster sum)."""
    medoids = []
    for g in sorted(np.unique(labels)):
        idx = np.where(labels == g)[0]
        sub = distance_matrix[np.ix_(idx, idx)]
        medoids.append(int(idx[int(np.argmin(sub.sum(axis=1)))]))
    return np.asarray(medoids, dtype=int)


def _cluster_size_stats(labels: np.ndarray) -> dict[str, Any]:
    counts = np.bincount(labels.astype(int))
    counts = counts[counts > 0]
    return {
        "sizes": [int(c) for c in counts],
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": float(counts.mean()),
        "std": float(counts.std(ddof=0)),
        "cv": float(counts.std(ddof=0) / counts.mean()) if counts.mean() else 0.0,
        "mega_share": float(counts.max() / counts.sum()),
        "orphan_count": int(np.sum(counts == 1)),
    }


def _centroid_separation(feature_matrix: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Pairwise L1 distances between arithmetic means of clusters in feature space."""
    centroids = []
    ids = sorted(np.unique(labels))
    for g in ids:
        centroids.append(feature_matrix[labels == g].mean(axis=0))
    C = np.vstack(centroids)
    D = pairwise_distances(C, metric="manhattan")
    tri = D[np.triu_indices(len(ids), k=1)]
    if len(tri) == 0:
        return {"min_centroid_l1": None, "mean_centroid_l1": None, "nearest_pair": None}
    i, j = np.unravel_index(np.argmin(D + np.eye(len(ids)) * 1e9), D.shape)
    return {
        "min_centroid_l1": float(tri.min()),
        "mean_centroid_l1": float(tri.mean()),
        "max_centroid_l1": float(tri.max()),
        "nearest_pair": (int(ids[i]), int(ids[j]), float(D[i, j])),
    }


def _feature_discrimination(feature_matrix: np.ndarray, labels: np.ndarray) -> list[dict[str, Any]]:
    """Between / within variance ratio per feature (higher = more split-driving)."""
    rows = []
    overall = feature_matrix.mean(axis=0)
    for j, name in enumerate(FEATURE_NAMES):
        col = feature_matrix[:, j]
        between = 0.0
        within = 0.0
        for g in np.unique(labels):
            mask = labels == g
            mu = col[mask].mean()
            between += mask.sum() * (mu - overall[j]) ** 2
            within += ((col[mask] - mu) ** 2).sum()
        # Cap infinite ratio (perfect within-cluster homogeneity) for JSON safety.
        ratio = float(between / within) if within > 1e-12 else 999.0
        rows.append(
            {
                "feature": name,
                "between_ss": float(between),
                "within_ss": float(within),
                "variance_ratio": ratio,
                "perfect_homogeneity": bool(within <= 1e-12),
            }
        )
    rows.sort(key=lambda r: r["variance_ratio"], reverse=True)
    return rows


def _top_feature_ranges(df: pd.DataFrame, labels: np.ndarray, top_n: int = 5) -> list[dict[str, Any]]:
    """Per-cluster defining ranges on raw Likert + dominant learning style."""
    out = []
    for g in sorted(np.unique(labels)):
        sub = df.loc[labels == g]
        ranges = []
        for col in RAW_SKILL_COLS:
            vals = sub[col].astype(float)
            ranges.append(
                {
                    "feature": col,
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=0)),
                }
            )
        style_mode = sub["Learning_Style"].mode().iloc[0]
        style_purity = float((sub["Learning_Style"] == style_mode).mean())
        ranges.sort(key=lambda r: -r["std"])  # tightest later; report all three + style
        out.append(
            {
                "cluster_id": int(g) + 1,
                "size": int(len(sub)),
                "skill_ranges": ranges,
                "dominant_learning_style": str(style_mode),
                "learning_style_purity": style_purity,
                "top_defining": [
                    {
                        "feature": r["feature"],
                        "range": f"{r['min']:.0f}–{r['max']:.0f}",
                        "mean": round(r["mean"], 2),
                        "std": round(r["std"], 3),
                    }
                    for r in sorted(ranges, key=lambda r: r["std"])[:top_n]
                ]
                + [
                    {
                        "feature": "Learning_Style",
                        "range": style_mode,
                        "mean": round(style_purity, 3),
                        "std": None,
                    }
                ],
            }
        )
    return out


def _demographic_association(df: pd.DataFrame, labels: np.ndarray) -> dict[str, Any]:
    """
    Association of ignored demographics with final labels (spurious correlation check).
    Not leakage into the model — Gender/Diversity are absent from X — but useful QA.
    """
    from sklearn.metrics import normalized_mutual_info_score

    result: dict[str, Any] = {}
    for col in ("Gender", "Diversity"):
        if col not in df.columns:
            continue
        codes, _ = pd.factorize(df[col].astype(str))
        result[col] = {
            "nmi_with_cluster": float(normalized_mutual_info_score(codes, labels)),
        }
    # Feature-side NMI baseline: Motivation discretized
    if "Motivation" in df.columns:
        result["Motivation_baseline_nmi"] = float(
            normalized_mutual_info_score(df["Motivation"].astype(int), labels)
        )
    return result


def _quality_bundle(
    feature_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    medoid_indices: np.ndarray | None,
) -> dict[str, Any]:
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        raise ValueError("Need 2 <= n_clusters < n_samples for unsupervised metrics.")

    sil_man = float(silhouette_score(distance_matrix, labels, metric="precomputed"))
    sil_euc = float(silhouette_score(feature_matrix, labels, metric="euclidean"))
    db = float(davies_bouldin_score(feature_matrix, labels))
    ch = float(calinski_harabasz_score(feature_matrix, labels))

    if medoid_indices is None:
        medoid_indices = _nearest_medoids(distance_matrix, labels)
    pam = _pam_cost(distance_matrix, labels, medoid_indices)

    # Mean within-cluster pairwise Manhattan (diagonal excluded)
    within_means = []
    for g in unique:
        idx = np.where(labels == g)[0]
        if len(idx) < 2:
            within_means.append(0.0)
            continue
        sub = distance_matrix[np.ix_(idx, idx)]
        within_means.append(float(sub[np.triu_indices(len(idx), k=1)].mean()))

    return {
        "silhouette_manhattan": sil_man,
        "silhouette_euclidean": sil_euc,
        "davies_bouldin_euclidean": db,
        "calinski_harabasz_euclidean": ch,
        "pam_total_cost": pam,
        "mean_within_cluster_pairwise_l1": float(np.mean(within_means)),
        "medoid_indices": [int(i) for i in medoid_indices],
    }


def diagnose(
    csv_path: str | Path,
    target_size: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    df = prepare_for_grouping(csv_path)
    X = compute_feature_vector(df)
    D = compute_psychometric_distance_matrix(X)
    n_groups = calculate_n_groups(len(df), target_size)

    # Raw PAM (pre size-balance) — true clustering objective
    labels_pam, medoids = kmedoids_pam(D, n_groups, random_state=random_state)
    quality_pam = _quality_bundle(X, D, labels_pam, medoids)

    # Production path (PAM + size balance)
    result = run_grouping_pipeline(df, target_size, random_state=random_state, backend="cpu")
    labels = result.labels
    medoids_final = _nearest_medoids(D, labels)
    quality_prod = _quality_bundle(X, D, labels, medoids_final)

    # Confirm demographics excluded from feature matrix
    assert X.shape[1] == 6, f"Expected 6 psychometric features, got {X.shape[1]}"
    leakage_guard = {
        "feature_dim": int(X.shape[1]),
        "feature_names": FEATURE_NAMES,
        "gender_in_features": False,
        "diversity_in_features": False,
        "note": "Gender/Diversity absent from compute_feature_vector by construction.",
    }

    sizes_pam = _cluster_size_stats(labels_pam)
    sizes_prod = _cluster_size_stats(labels)

    report = {
        "input": str(csv_path),
        "n_students": int(len(df)),
        "target_size": target_size,
        "n_groups": n_groups,
        "random_state": random_state,
        "leakage_guard": leakage_guard,
        "pre_balance_pam": {
            "quality": quality_pam,
            "sizes": sizes_pam,
        },
        "production_balanced": {
            "quality": quality_prod,
            "sizes": sizes_prod,
            "group_size_range": result.group_size_range,
            "backend": result.backend_label,
        },
        "balance_delta": {
            "silhouette_manhattan": quality_prod["silhouette_manhattan"]
            - quality_pam["silhouette_manhattan"],
            "pam_total_cost": quality_prod["pam_total_cost"] - quality_pam["pam_total_cost"],
            "davies_bouldin_euclidean": quality_prod["davies_bouldin_euclidean"]
            - quality_pam["davies_bouldin_euclidean"],
            "calinski_harabasz_euclidean": quality_prod["calinski_harabasz_euclidean"]
            - quality_pam["calinski_harabasz_euclidean"],
        },
        "centroid_separation_l1": _centroid_separation(X, labels),
        "feature_discrimination": _feature_discrimination(X, labels),
        "skill_variance": compute_skill_variance(df, labels),
        "per_cluster_ranges": _top_feature_ranges(df, labels),
        "demographic_association": _demographic_association(df, labels),
        "anomalies": [],
    }

    # Anomaly flags
    if sizes_prod["orphan_count"]:
        report["anomalies"].append(
            f"Orphan clusters (size=1): {sizes_prod['orphan_count']}"
        )
    if sizes_prod["mega_share"] >= 0.5:
        report["anomalies"].append(
            f"Mega-cluster share {sizes_prod['mega_share']:.0%} of roster"
        )
    if quality_prod["silhouette_manhattan"] < 0.1:
        report["anomalies"].append(
            f"Weak Manhattan silhouette ({quality_prod['silhouette_manhattan']:.3f}); "
            "clusters may overlap heavily after size balancing"
        )
    if report["balance_delta"]["pam_total_cost"] > 0.15 * max(quality_pam["pam_total_cost"], 1e-9):
        report["anomalies"].append(
            "Size balancing raised PAM cost >15% vs raw K-Medoids "
            f"(d={report['balance_delta']['pam_total_cost']:.3f})"
        )
    nearest = report["centroid_separation_l1"].get("nearest_pair")
    if nearest and nearest[2] < 1.0:
        report["anomalies"].append(
            f"Tight centroid pair groups {nearest[0]+1} & {nearest[1]+1} "
            f"(L1={nearest[2]:.3f}) — possible overlap zone"
        )
    dem = report["demographic_association"]
    mot_nmi = dem.get("Motivation_baseline_nmi", 0.0)
    for col in ("Gender", "Diversity"):
        if col in dem and dem[col]["nmi_with_cluster"] > mot_nmi + 0.05:
            report["anomalies"].append(
                f"{col} NMI with clusters ({dem[col]['nmi_with_cluster']:.3f}) "
                f"exceeds Motivation baseline ({mot_nmi:.3f}) — review for spurious structure"
            )

    # Medoid accuracy: distance from arithmetic mean to nearest assigned medoid in feature space
    medoid_checks = []
    for g, mid in zip(sorted(np.unique(labels)), medoids_final):
        members = X[labels == g]
        geom_mean = members.mean(axis=0)
        medoid_vec = X[mid]
        medoid_checks.append(
            {
                "cluster_id": int(g) + 1,
                "medoid_row": int(mid),
                "l1_mean_to_medoid": float(np.abs(geom_mean - medoid_vec).sum()),
                "is_member": bool(labels[mid] == g),
            }
        )
    report["medoid_accuracy"] = medoid_checks

    return report


def _print_report(report: dict[str, Any]) -> None:
    q = report["production_balanced"]["quality"]
    qp = report["pre_balance_pam"]["quality"]
    print("=" * 72)
    print("FEATURE-ONLY CLUSTER HEALTH DIAGNOSTIC")
    print("=" * 72)
    print(f"Input: {report['input']}")
    print(
        f"N={report['n_students']} | k={report['n_groups']} | "
        f"target_size={report['target_size']} | seed={report['random_state']}"
    )
    print(f"Sizes (production): {report['production_balanced']['sizes']['sizes']} "
          f"({report['production_balanced']['group_size_range']})")
    print(f"Sizes (raw PAM):    {report['pre_balance_pam']['sizes']['sizes']}")
    print()
    print("QUALITY (production / balanced)")
    print(f"  Silhouette (Manhattan, precomputed): {q['silhouette_manhattan']:.4f}")
    print(f"  Silhouette (Euclidean, features):    {q['silhouette_euclidean']:.4f}")
    print(f"  Davies-Bouldin (Euclidean):          {q['davies_bouldin_euclidean']:.4f}  (lower better)")
    print(f"  Calinski-Harabasz (Euclidean):       {q['calinski_harabasz_euclidean']:.4f}  (higher better)")
    print(f"  PAM total cost (L1 to medoids):      {q['pam_total_cost']:.4f}")
    print(f"  Mean within-cluster pairwise L1:     {q['mean_within_cluster_pairwise_l1']:.4f}")
    print()
    print("vs RAW PAM (before size balance)")
    print(
        f"  Silhouette Man: {qp['silhouette_manhattan']:.4f} -> {q['silhouette_manhattan']:.4f} "
        f"(d {report['balance_delta']['silhouette_manhattan']:+.4f})"
    )
    print(
        f"  PAM cost:       {qp['pam_total_cost']:.4f} -> {q['pam_total_cost']:.4f} "
        f"(d {report['balance_delta']['pam_total_cost']:+.4f})"
    )
    print(
        f"  Davies-Bouldin: {qp['davies_bouldin_euclidean']:.4f} -> {q['davies_bouldin_euclidean']:.4f} "
        f"(d {report['balance_delta']['davies_bouldin_euclidean']:+.4f})"
    )
    print(
        f"  Calinski-Harabasz: {qp['calinski_harabasz_euclidean']:.4f} -> "
        f"{q['calinski_harabasz_euclidean']:.4f} "
        f"(d {report['balance_delta']['calinski_harabasz_euclidean']:+.4f})"
    )
    print()
    print("FEATURE DISCRIMINATION (between/within SS ratio)")
    for row in report["feature_discrimination"]:
        print(f"  {row['feature']:18s}  ratio={row['variance_ratio']:.3f}")
    print()
    print("DEMOGRAPHIC ASSOCIATION (NMI; not used in model)")
    for k, v in report["demographic_association"].items():
        if isinstance(v, dict):
            print(f"  {k}: {v['nmi_with_cluster']:.4f}")
        else:
            print(f"  {k}: {v:.4f}")
    print()
    cs = report["centroid_separation_l1"]
    print(
        f"Centroid L1 separation: min={cs['min_centroid_l1']:.3f} "
        f"mean={cs['mean_centroid_l1']:.3f} max={cs['max_centroid_l1']:.3f}"
    )
    if report["anomalies"]:
        print()
        print("ANOMALIES")
        for a in report["anomalies"]:
            print(f"  ! {a}")
    else:
        print()
        print("ANOMALIES: none flagged")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose feature-only cluster health")
    parser.add_argument("--csv", default=str(DEFAULT_TEMPLATE_CSV), help="Input CSV path")
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write full JSON report",
    )
    args = parser.parse_args()

    report = diagnose(args.csv, target_size=args.group_size, random_state=args.seed)

    out_dir = Path("backend/output/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(args.csv).stem
    default_json = out_dir / f"feature_cluster_health_{stem}_{stamp}.json"
    json_path = Path(args.json_out) if args.json_out else default_json
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _print_report(report)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
