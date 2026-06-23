"""
Research-only clustering comparison (not classroom grouping).

Compares K-Means and K-Medoids variants (Euclidean, Manhattan, optional Gower)
on the template dataset. Uses ``prepare_for_grouping`` for ingest but does
**not** call ``run_grouping_pipeline``.

Outputs: ``backend/output_plots/runs/<timestamp>_<uuid>/`` (CSVs, PNGs).

Usage:
    python -m backend.run_full_evaluation

For production grouping see ``generate_groups.py`` / ``api.py``.
See ``docs/ARCHITECTURE.md``.
"""

import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from .paths import DEFAULT_EVAL_OUTPUT_DIR, DEFAULT_TEMPLATE_CSV
from .data_loader import prepare_for_grouping
from .group_config import calculate_n_groups
from .clustering import (
    compute_feature_vector, kmeans_custom, visualize_clustering, enforce_group_size, compute_distance_matrix
)
from .kmedoids import kmedoids_pam
from .evaluate_clustering import (
    evaluate_clustering, print_evaluation_report, create_comparison_table,
    plot_metrics_comparison, plot_cluster_balance
)
from .skill_variance import compute_skill_variance, print_skill_variance_report, plot_skill_variance


def run_full_evaluation(
    n_clusters: int = None,
    group_size: int = 5,
    random_state: int = 42,
    visualize: bool = True,
    save_plots: bool = True
):
    """
    Run complete clustering evaluation with K-Means and K-Medoids.

    Args:
        n_clusters: Number of clusters to form
        group_size: Target group size (will enforce balanced groups)
        random_state: Random seed for reproducibility
        visualize: Whether to show visualization plots
        save_plots: Whether to save plots to backend/output_plots/

    Returns:
        Dictionary with all evaluation results
    """
    print("\n" + "="*80)
    print("FULL CLUSTERING EVALUATION PIPELINE")
    print("="*80)

    # Per-run output so concurrent evaluations do not overwrite
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_id = f"{stamp}_{uuid.uuid4().hex[:8]}"
    output_dir = DEFAULT_EVAL_OUTPUT_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n Evaluation outputs: {output_dir}")

    # Same ingest path as production API/CLI
    print("\n Loading and preprocessing student data...")
    data_file = DEFAULT_TEMPLATE_CSV
    df = prepare_for_grouping(str(data_file))
    n_students = len(df)
    print(f"  Loaded {len(df)} students")

    if n_clusters is None:
        n_clusters = calculate_n_groups(n_students, group_size)
        if n_clusters < 2 and n_students >= 4:
            n_clusters = 2

    print("\n Computing feature vectors...")
    feature_matrix = compute_feature_vector(df)
    print(f"  Feature matrix shape: {feature_matrix.shape}")

    # Compute all distance matrices for K-Medoids/Gower and K-Medoids/Manhattan
    dist_euclidean, dist_manhattan, dist_gower = compute_distance_matrix(df, feature_matrix)
 
    # Run clusterings and evaluate
    print("\n Running clusterings...")
    results = []

    #=================================#
    # 1. K-Means Euclidean
    #=================================#
    #Cluster
    print("\n  K-Means (Euclidean)...")
    labels_kmeans_euc, centroids_euc = kmeans_custom(
        feature_matrix, n_clusters, random_state=random_state, return_centroids=True, metric="euclidean"
    )

    #Enforce Group Size
    labels_kmeans_euc = enforce_group_size(
        labels_kmeans_euc,
        group_size,
        feature_matrix=feature_matrix,
        metric="euclidean",
    )

    #calculate skill variance for this clustering
    skill_var_euc = compute_skill_variance(df, labels_kmeans_euc)

    #Evaluate Clustering
    result_kmeans_euc = evaluate_clustering(
        feature_matrix, 
        labels_kmeans_euc, 
        df, 
        metric_name="Euclidean", 
        algorithm_name="K-Means"
    )

    # Add skill variance to results
    result_kmeans_euc['skill_variance'] = skill_var_euc

    # Append and report
    results.append(result_kmeans_euc)
    print_evaluation_report(result_kmeans_euc)
    print_skill_variance_report(skill_var_euc, "K-Means", "Euclidean")

    #visualize clustering 
    if visualize:
        visualize_clustering(feature_matrix, labels_kmeans_euc, n_clusters, df=df, metric="euclidean")


    #=================================#
    # 2. K-Means Manhattan
    #=================================#
    #Cluster
    print("\n  K-Means (Manhattan)...")
    labels_kmeans_man, centroids_man = kmeans_custom(
        feature_matrix, n_clusters, random_state=random_state, return_centroids=True, metric="manhattan"
    )

    #Enforce Group Size
    labels_kmeans_man = enforce_group_size(
        labels_kmeans_man, 
        group_size,
        feature_matrix=feature_matrix,
        metric="manhattan"
    )

    #calculate skill variance for this clustering
    skill_var_man = compute_skill_variance(df, labels_kmeans_man)

    #Evaluate Clustering
    result_kmeans_man = evaluate_clustering(
        feature_matrix, 
        labels_kmeans_man, 
        df, 
        metric_name="Manhattan", 
        algorithm_name="K-Means"
    )

    # Add skill variance to results
    result_kmeans_man['skill_variance'] = skill_var_man

    # Append and report
    results.append(result_kmeans_man)
    print_evaluation_report(result_kmeans_man)
    print_skill_variance_report(skill_var_man, "K-Means", "Manhattan")

    if visualize:
        visualize_clustering(feature_matrix, labels_kmeans_man, n_clusters, df=df, metric="manhattan")

    #=================================#
    # 3. K-Medoids Manhattan
    #=================================#
    #Cluster
    print("\n  K-Medoids (Manhattan)...")
    labels_kmedoids_man, medoid_indices = kmedoids_pam(
        dist_manhattan, n_clusters,random_state=random_state
    )

    #Enforce Group Size
    labels_kmedoids_man = enforce_group_size(
        labels_kmedoids_man,
        group_size,
        distance_matrix=dist_manhattan,
    )

    # calculate skill variance for this clustering
    skill_var_kmedoids = compute_skill_variance(df, labels_kmedoids_man)

    #Evaluate Clustering
    result_kmedoids_man = evaluate_clustering(
        feature_matrix, 
        labels_kmedoids_man, 
        df, 
        metric_name="Manhattan", 
        algorithm_name="K-Medoids"
    )

    # Add skill variance to results
    result_kmedoids_man['skill_variance'] = skill_var_kmedoids


    # Append and report
    results.append(result_kmedoids_man)
    print(f"  Medoid indices: {medoid_indices}")
    print_evaluation_report(result_kmedoids_man)
    print_skill_variance_report(skill_var_kmedoids, "K-Medoids", "Manhattan")

    # Visualize clustering
    if visualize:
        visualize_clustering(feature_matrix, labels_kmedoids_man, n_clusters, df=df, metric="manhattan")
    

    #=================================#
    # 4. K-Medoids Gower
    #=================================#
    if dist_gower is None:
        print("\n Skipping K-Medoids (Gower): install gower package (pip install gower)")
    else:
        print("\n K-Medoids (Gower Distance)...")

        labels_kmedoids_gower, medoid_indices_gower = kmedoids_pam(
            dist_gower, n_clusters, random_state=random_state
        )

        labels_kmedoids_gower = enforce_group_size(
            labels_kmedoids_gower,
            group_size,
            distance_matrix=dist_gower,
        )

        skill_var_gower = compute_skill_variance(df, labels_kmedoids_gower)

        result_kmedoids_gower = evaluate_clustering(
            feature_matrix,
            labels_kmedoids_gower,
            df,
            metric_name="Gower",
            algorithm_name="K-Medoids",
        )

        result_kmedoids_gower["silhouette_score"] = silhouette_score(
            dist_gower, labels_kmedoids_gower, metric="precomputed"
        )
        result_kmedoids_gower["skill_variance"] = skill_var_gower

        results.append(result_kmedoids_gower)
        print(f"  Medoid indices: {medoid_indices_gower}")
        print_evaluation_report(result_kmedoids_gower)
        print_skill_variance_report(skill_var_gower, "K-Medoids", "Gower")

        if visualize:
            visualize_clustering(
                feature_matrix, labels_kmedoids_gower, n_clusters, df=df, metric="Gower"
            )



    #=================================#
    # Generate Comparison Tables & Visualizations
    #=================================#
  
    # Generate comparison table
    print("\n[4/4] Generating comparison tables and visualizations...")
    comparison_df = create_comparison_table(results)
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print()

    # Save comparison table to CSV
    if save_plots:
        comparison_csv = output_dir / "clustering_comparison.csv"
        comparison_df.to_csv(comparison_csv, index=False)
        print(f"Comparison table saved to: {comparison_csv}")

    # Plot metrics comparison
    metrics_plot_path = output_dir / "metrics_comparison.png" if save_plots else None
    plot_metrics_comparison(results, save_path=metrics_plot_path)

    # Plot skill variance comparison
    skill_variance_plot_path = output_dir / "skill_variance_comparison.png" if save_plots else None
    plot_skill_variance(results, save_path=skill_variance_plot_path)

    # Plot cluster balance
    balance_plot_path = output_dir / "cluster_balance.png" if save_plots else None
    plot_cluster_balance(results, save_path=balance_plot_path)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Total evaluations: {len(results)}")
    print(f"Best Silhouette Score: {max(r['silhouette_score'] for r in results):.4f}")
    print()

    return {
        "results": results,
        "comparison_table": comparison_df,
        "dataframe": df,
        "feature_matrix": feature_matrix,
    }

def get_user_input():
    """Handles CLI input for group size."""
    print("\n--- GroupGen Evaluation ---")
    
    while True:
        user_in = input("\nEnter desired group size: ").strip()
        
        if not user_in:
            return 10 # Default
            
        try:
            size = int(user_in)
            if size <= 1 or size > 100:
                print("Please enter a size between 2 and 100.")
                continue
            return size
        except ValueError:
            print("Invalid input. Please enter a number.")


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit(
            "Run from the repo root:  python -m backend.run_full_evaluation"
        )
    target_group_size = get_user_input()
    run_full_evaluation(group_size=target_group_size)
