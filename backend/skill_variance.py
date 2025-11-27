"""
Skill Variance Analysis for Group Formation

For many classroom tasks you may prefer groups where students are similar
(homogeneous) on motivation, work ethic and self-esteem so they can work
at a similar pace and expectations. In that case, LOWER variance in those
skills inside a group is preferable.

Skill variance here still measures the spread of Motivation, Work_Ethic and
Self_Esteem inside each cluster. But the interpretation is inverted: lower
mean variance → more homogeneous group composition → often better for
tasks requiring tight coordination or equal participation.

Formula: mean(var(Motivation), var(Work_Ethic), var(Self_Esteem)) per cluster
         Lower = Better when you want similar students grouped together
"""

import numpy as np
import pandas as pd
from typing import Dict
import matplotlib.pyplot as plt


def compute_skill_variance(df, labels):
    """
    Compute skill variance (Motivation, Work_Ethic, Self_Esteem) for each cluster.

    For group formation, LOWER variance is better when you prefer students in
    each group to be similar (homogeneous) on these skill dimensions.
    Args:
        df: DataFrame with student attributes
        labels: Cluster labels for each student

    Returns:
        Dictionary with:
        - by_cluster: per-cluster skill variance breakdown
        - overall_mean_variance: average variance across all clusters
        - overall_std_variance: standard deviation of cluster variances
    """
    skill_cols = ['Motivation', 'Work_Ethic', 'Self_Esteem']
    
    # Check that required columns exist
    missing = [col for col in skill_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    cluster_variances = {}
    all_variances = []

    for cluster_id in sorted(np.unique(labels)):
        cluster_mask = labels == cluster_id
        cluster_df = df[cluster_mask]

        # Compute variance of each skill in this cluster
        skill_vars = cluster_df[skill_cols].var()
        mean_var = skill_vars.mean()

        cluster_variances[int(cluster_id)] = {
            'motivation_variance': float(skill_vars['Motivation']) if not np.isnan(skill_vars['Motivation']) else 0.0,
            'work_ethic_variance': float(skill_vars['Work_Ethic']) if not np.isnan(skill_vars['Work_Ethic']) else 0.0,
            'self_esteem_variance': float(skill_vars['Self_Esteem']) if not np.isnan(skill_vars['Self_Esteem']) else 0.0,
            'mean_skill_variance': float(mean_var) if not np.isnan(mean_var) else 0.0,
        }
        all_variances.append(mean_var)

    return {
        'by_cluster': cluster_variances,
        'overall_mean_variance': float(np.mean(all_variances)),
        'overall_std_variance': float(np.std(all_variances)),
    }


def print_skill_variance_report(skill_var_result, algorithm, metric):
    """
    Pretty-print skill variance analysis.

    Args:
        skill_var_result: Dictionary from compute_skill_variance()
        algorithm: Algorithm name (e.g., "K-Means")
        metric: Distance metric (e.g., "Euclidean")
    """
    print("\n" + "="*80)
    print(f"SKILL VARIANCE ANALYSIS ({algorithm}, {metric})")
    print("="*80)
    print("Lower variance = Better for group formation (more similar students)\n")

    print(f"Overall Mean Skill Variance: {skill_var_result['overall_mean_variance']:.4f}")
    print(f"Variance Std Dev (consistency across groups): {skill_var_result['overall_std_variance']:.4f}")
    print()

    print("PER-CLUSTER SKILL VARIANCE:")
    print("-" * 80)
    for cluster_id, variances in skill_var_result['by_cluster'].items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Motivation Variance:  {variances['motivation_variance']:.4f}")
        print(f"  Work_Ethic Variance:  {variances['work_ethic_variance']:.4f}")
        print(f"  Self_Esteem Variance: {variances['self_esteem_variance']:.4f}")
        print(f"  ─────────────────────────────")
    print(f"  Mean Skill Variance:  {variances['mean_skill_variance']:.4f} ← Group Homogeneity Score (lower = more similar)")
    print()


def plot_skill_variance(results, save_path=None):
    """
    Visualize skill variance across clusterings.

    Args:
        results: List of dicts containing 'algorithm', 'metric', 'skill_variance' keys
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Overall mean variance per algorithm/metric
    ax = axes[0]
    names = [f"{r['algorithm']}\n({r['metric']})" for r in results]
    overall_vars = [r['skill_variance']['overall_mean_variance'] for r in results]
    std_vars = [r['skill_variance']['overall_std_variance'] for r in results]
    # Color coding: lower variance = green (good), higher = red (bad)
    colors = ['green' if v < 0.3 else 'orange' if v < 0.5 else 'red' for v in overall_vars]
    bars = ax.bar(names, overall_vars, yerr=std_vars, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Mean Skill Variance (lower = better)', fontsize=11)
    ax.set_title('Overall Skill Variance (↓ Lower = Better for Groups)', fontsize=12, fontweight='bold')
    ax.axhline(y=0.3, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Good (<0.3)')
    ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Fair (<0.5)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, var in zip(bars, overall_vars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{var:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 2: Per-cluster variance heatmap
    ax = axes[1]
    all_cluster_ids = []
    all_variances_list = []
    algo_metric_labels = []

    for result in results:
        for cluster_id, var_dict in result['skill_variance']['by_cluster'].items():
            all_cluster_ids.append(cluster_id)
            all_variances_list.append(var_dict['mean_skill_variance'])
            algo_metric_labels.append(f"{result['algorithm']}\n({result['metric']})")

    # Create scatter plot of per-cluster variance
    unique_labels = sorted(set(algo_metric_labels))
    colors_map = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    color_dict = {label: colors_map[i] for i, label in enumerate(unique_labels)}

    for label in unique_labels:
        mask = [l == label for l in algo_metric_labels]
        x_pos = np.where(mask)[0]
        y_vals = [v for i, v in enumerate(all_variances_list) if mask[i]]
        ax.scatter(x_pos, y_vals, s=100, label=label, alpha=0.7, edgecolors='black')

    ax.set_xlabel('Cluster Index', fontsize=11)
    ax.set_ylabel('Mean Skill Variance per Cluster', fontsize=11)
    ax.set_title('Per-Cluster Skill Variance Distribution', fontsize=12, fontweight='bold')
    ax.axhline(y=0.5, color='green', linestyle='--', linewidth=1, alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')

    plt.tight_layout()

    # if save_path:
    #     plt.savefig(save_path, dpi=150, bbox_inches='tight')
    #     print(f"Skill variance plot saved to: {save_path}")S

    plt.show()
