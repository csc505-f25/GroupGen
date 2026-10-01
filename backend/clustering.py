"""
Feature engineering, distance matrices, and post-clustering adjustments.

Production path (via ``pipeline.run_grouping_pipeline``):
  - ``compute_feature_vector`` — Motivation, Self_Esteem, Work_Ethic, Learning_Style only
  - ``compute_psychometric_distance_matrix`` — Manhattan for clustering and size balancing
  - ``enforce_group_size`` — balance counts after K-Medoids by moving the student
    closest (Manhattan) to the undersized group's medoid; reopens empty clusters
    so identical-profile students still split to the target size
  - ``check_*_isolation`` / ``rebalance_demographic_column`` — legacy research helpers (not used by production pipeline on ``feature-only-clustering``)

Research / legacy (not used for live classroom grouping):
  - ``compute_distance_matrix`` — also builds optional Gower matrix (includes demographics)
  - ``kmeans_custom``, ``form_balanced_groups``, ``visualize_*`` — evaluation and plots

See ``docs/ARCHITECTURE.md`` for the full lifecycle.
"""

import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

VALID_LEARNING_STYLES = ["Visual", "Auditory", "Kinesthetic"]
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from .group_config import calculate_n_groups
from .kmedoids import kmedoids_pam

# ==========================================
# 1. Feature Engineering & Distances
# ==========================================

def _cluster_medoid_index(
    distance_matrix: np.ndarray, member_indices: np.ndarray
) -> int:
    """
    Index of the cluster medoid under the given distance matrix.

    The medoid minimizes total distance to other members of the same cluster
    (same definition as K-Medoids). Ties break toward the smallest row index
    via ``argmin``.
    """
    members = np.asarray(member_indices, dtype=int)
    if members.size == 0:
        raise ValueError("Cannot compute medoid of an empty cluster.")
    if members.size == 1:
        return int(members[0])
    sub = distance_matrix[np.ix_(members, members)]
    return int(members[int(np.argmin(sub.sum(axis=1)))])


def enforce_group_size(
    labels: np.ndarray,
    group_size: int,
    *,
    distance_matrix: np.ndarray = None,
    feature_matrix: np.ndarray = None,
    metric: str = "manhattan",
) -> np.ndarray:
    """
    Redistribute students so each group matches the ceil(n/k) size distribution.

    Target counts: ``base = n // k`` and ``base + 1`` for the remainder groups.
    ``k`` is ``calculate_n_groups(n, group_size)``, not the number of non-empty
    labels. If PAM leaves a cluster empty (common when many students share the
    same profile), those missing groups are reopened and seeded so no team
    exceeds ``group_size``.

    Production passes the psychometric Manhattan ``distance_matrix`` (same matrix
    used by K-Medoids). Each move picks the oversized-group student closest to
    the **medoid** of an undersized group — not nearest neighbor to any member,
    and not Gender/Diversity. Empty receivers are seeded first (no medoid yet).

    Evaluation may instead pass ``feature_matrix`` (distance to arithmetic mean).

    Raises ``ValueError`` if balancing cannot complete within ``2 * n`` moves.
    """
    if len(labels) == 0:
        return labels

    n_students = len(labels)
    intended_n_groups = calculate_n_groups(n_students, group_size)
    # Compact to 0..n_present-1 so unused PAM ids become trailing empty slots.
    _, compact = np.unique(labels, return_inverse=True)
    new_labels = compact.astype(int, copy=True)
    n_present = int(new_labels.max()) + 1
    # Never drop below the classroom group count; keep extras if a caller
    # already created more non-empty clusters than the formula requires.
    n_groups = max(intended_n_groups, n_present)
    group_ids = list(range(n_groups))

    # Example: 31 students, 7 groups → base=4, remainder=3 → three groups of 5, four of 4.
    base_size = n_students // n_groups
    remainder = n_students % n_groups
    current_counts = {g: int(np.sum(new_labels == g)) for g in group_ids}

    # Give the extra +1 seats to the groups that are currently largest (deterministic).
    sorted_groups_by_curr_size = sorted(
        group_ids, key=lambda g: current_counts[g], reverse=True
    )
    target_sizes = {
        g_id: base_size + (1 if i < remainder else 0)
        for i, g_id in enumerate(sorted_groups_by_curr_size)
    }

    max_iter = n_students * 2  # hard stop — cannot infinite-loop
    # Cheaper than any real distance (>= 0) so empty groups fill before
    # students move between already-populated teams.
    empty_seed_cost = -1.0

    for iteration in range(max_iter):
        donors = [g for g in group_ids if current_counts[g] > target_sizes[g]]
        receivers = [g for g in group_ids if current_counts[g] < target_sizes[g]]

        if not donors or not receivers:
            break

        best_move = None
        min_cost = float("inf")

        if distance_matrix is not None:
            # Production: move the donor student closest to the receiver medoid (Manhattan).
            receiver_medoids: dict = {}
            for r_id in receivers:
                receiver_indices = np.where(new_labels == r_id)[0]
                if receiver_indices.size == 0:
                    receiver_medoids[r_id] = None
                else:
                    receiver_medoids[r_id] = _cluster_medoid_index(
                        distance_matrix, receiver_indices
                    )

            for d_id in donors:
                for student_idx in np.where(new_labels == d_id)[0]:
                    for r_id, medoid_idx in receiver_medoids.items():
                        if medoid_idx is None:
                            cost = empty_seed_cost
                        else:
                            cost = float(distance_matrix[student_idx, medoid_idx])
                        if np.isfinite(cost) and cost < min_cost:
                            min_cost = cost
                            best_move = (int(student_idx), r_id)
        elif feature_matrix is not None:
            for d_id in donors:
                donor_indices = np.where(new_labels == d_id)[0]
                donor_features = feature_matrix[donor_indices]
                for r_id in receivers:
                    mask = new_labels == r_id
                    if not np.any(mask):
                        cost = empty_seed_cost
                        if cost < min_cost:
                            min_cost = cost
                            best_move = (int(donor_indices[0]), r_id)
                        continue
                    center = feature_matrix[mask].mean(axis=0).reshape(1, -1)
                    dists = pairwise_distances(
                        donor_features, center, metric=metric
                    ).flatten()
                    local_min_idx = int(np.argmin(dists))
                    local_min_dist = float(dists[local_min_idx])
                    if local_min_dist < min_cost:
                        min_cost = local_min_dist
                        best_move = (int(donor_indices[local_min_idx]), r_id)
        else:
            d_id, r_id = donors[0], receivers[0]
            best_move = (int(np.where(new_labels == d_id)[0][0]), r_id)

        if best_move is None:
            raise ValueError(
                "Could not balance group sizes: no valid student move found. "
                "Try a different target group size or check your data."
            )

        student_to_move, new_group = best_move
        old_group = new_labels[student_to_move]
        new_labels[student_to_move] = new_group
        current_counts[old_group] -= 1
        current_counts[new_group] += 1

    still_imbalanced = any(
        current_counts[g] != target_sizes[g] for g in group_ids
    )
    if still_imbalanced:
        actual = {int(g): current_counts[g] for g in group_ids}
        expected = {int(g): target_sizes[g] for g in group_ids}
        raise ValueError(
            f"Could not balance group sizes for target {group_size} "
            f"after {max_iter} moves. Actual sizes: {actual}. Expected: {expected}."
        )

    return new_labels
    

def compute_feature_vector(df):
    """
    Build the N×M matrix used for production clustering (psychometric only).

    Columns used: Motivation, Self_Esteem, Work_Ethic (standardized) plus
    one-hot Learning_Style. Gender and Diversity are intentionally excluded.
    """
    # Scale 1–4 columns so no single trait dominates L1 distance.
    numeric_scaled = StandardScaler().fit_transform(
        df[["Motivation", "Self_Esteem", "Work_Ethic"]].values
    )
    # Fixed category order → stable columns across runs.
    style_encoder = OneHotEncoder(
        categories=[VALID_LEARNING_STYLES],
        sparse_output=False,
    )
    learning_style_encoded = style_encoder.fit_transform(df[["Learning_Style"]])
    return np.hstack([numeric_scaled, learning_style_encoded])


def compute_psychometric_distance_matrix(feature_matrix: np.ndarray) -> np.ndarray:
    """Manhattan distances from psychometric features only (production clustering)."""
    return pairwise_distances(feature_matrix, metric="manhattan")


def compute_distance_matrix(df, feature_matrix):
    """
    Distance matrices for clustering and research evaluation.

    Production pipeline uses ``compute_psychometric_distance_matrix`` only.
    Gower (includes Gender/Diversity) is for evaluation scripts, not live grouping.
    """
    eucdistance_matrix = pairwise_distances(feature_matrix, metric="euclidean")
    mandistance_matrix = compute_psychometric_distance_matrix(feature_matrix)

    # --- Research only: Gower mixes demographics into distance (NOT used in pipeline.py) ---
    gower_cols = ['Motivation', 'Self_Esteem', 'Work_Ethic', 'Gender', 'Diversity', 'Learning_Style']
    gower_data = df[gower_cols].copy()

    #Which columns are categorical
    categorical_cols = [gower_data[col].dtype == 'object' for col in gower_cols]

    # Gower is only needed for evaluation tournaments (optional dependency)
    try:
        import gower as gower_lib
        gower_distance_matrix = gower_lib.gower_matrix(gower_data, cat_features=categorical_cols)
    except ImportError:
        gower_distance_matrix = None

    return eucdistance_matrix, mandistance_matrix, gower_distance_matrix

# ==========================================
# 2. Clustering Algorithms
# ==========================================

def kmeans_custom(
    x: np.ndarray, 
    K: int, random_state: Optional[int] = None, 
    max_iter: int = 300, 
    return_centroids: bool = False, 
    metric: str = 'euclidean'
    ):

    """
    Custom K-Means clustering implementation.
    
    Args:
        x: NxM feature matrix (N students, M features)
        K: Number of clusters
        random_state: Random seed for reproducibility
        max_iter: Maximum number of iterations
        
    Returns:
        Array of cluster labels (0 to K-1) for each student
        If return_centroids=True, also returns centroids array
    """
    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
    
    # Random initialization of centroids
    idxs = np.random.choice(x.shape[0], K, replace=False)
    centroids = np.atleast_2d(x[idxs].copy())  # ensures centroids is always 2D
    prev = np.full(x.shape[0], -1, dtype=int)

    for iteration in range(max_iter):
        # assignment step: compute distances from each point to each centroid
        # support euclidean (L2) and manhattan (L1)
        if metric == 'euclidean':
            distances = np.sqrt(((x[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        elif metric == 'manhattan':
            distances = np.abs(x[:, np.newaxis, :] - centroids[np.newaxis, :, :]).sum(axis=2)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        C = np.argmin(distances, axis=1)
        
        # centroid update
        new_centroids = np.array([x[C==k].mean(axis=0) if np.any(C==k) else centroids[k]
                                for k in range(K)])
        
        # check for convergence
        if np.array_equal(C, prev):
            break
        prev = C.copy()
        centroids = new_centroids

    
    if return_centroids:
        return C, centroids
    return C


def initial_clustering(
    feature_matrix: np.ndarray,
    n_clusters: int,
    random_state: Optional[int] = None,
    visualize: bool = False,
    df: Optional[pd.DataFrame] = None
):
    """
    Perform initial clustering using custom K-Means algorithm.
    
    Args:
        feature_matrix: NxM feature matrix (N students, M features)
        n_clusters: Number of groups to form
        random_state: Random seed for reproducibility
        visualize: Whether to visualize the clustering results
        df: Optional DataFrame with student data (for visualization)
        

    Returns:
        Array of cluster labels (0 to n_clusters-1) for each student
    """
    # Use custom K-Means implementation (defaults to euclidean)
    labels, centroids = kmeans_custom(feature_matrix, n_clusters, random_state=random_state, return_centroids=True)
    
    # Visualize clustering if requested
    if visualize:
        visualize_clustering(feature_matrix, labels, n_clusters, df)
        if df is not None:
            visualize_feature_pairs(df, labels, n_clusters)
        visualize_clustering_with_centroids(feature_matrix, labels, n_clusters, centroids)
    
    return labels

def compare_metrics_and_visualize(
    df: pd.DataFrame, 
    desired_group_size: int, # Changed from n_clusters
    random_state: Optional[int] = 42, 
    use_kmedoids: bool = True):
    
    feature_matrix = compute_feature_vector(df)
    
    # --- DYNAMIC K CALCULATION ---
    n_students = len(df)
    n_clusters = max(1, n_students // desired_group_size)

    for metric in ("euclidean", "manhattan"):
        print(f"\n=== Running clustering with metric: {metric} (Calculated K={n_clusters}) ===")

        if metric == "manhattan" and use_kmedoids:
            # K-Medoids expects an N×N distance matrix, not a feature matrix.
            dist = compute_psychometric_distance_matrix(feature_matrix)
            labels, medoid_indices = kmedoids_pam(
                dist, n_clusters, random_state=random_state
            )
            centroids = feature_matrix[medoid_indices]
            print(f"Using K-Medoids (medoid indices: {medoid_indices})")
        else:
            labels, centroids = kmeans_custom(
                feature_matrix, n_clusters, random_state=random_state,
                return_centroids=True, metric=metric
            )
            print("Using K-Means")

        if desired_group_size is not None:
            if metric == "manhattan" and use_kmedoids:
                labels = enforce_group_size(
                    labels, desired_group_size, distance_matrix=dist
                )
            else:
                labels = enforce_group_size(
                    labels, desired_group_size, feature_matrix=feature_matrix, metric=metric
                )
            print(f"Groups forcibly balanced to {desired_group_size} students each.")
        visualize_clustering(feature_matrix, labels, n_clusters, df=df, metric=metric)


def visualize_clustering(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    df: Optional[pd.DataFrame] = None,
    metric: str = "euclidean"
):
    """
    Visualize clustering results using PCA to reduce dimensions to 2D.
    
    Args:
        feature_matrix: NxM feature matrix
        labels: Cluster labels for each student
        n_clusters: Number of clusters
        df: Optional DataFrame with student data (for displaying names)
    """
    # Minimal PCA-based visualization (single scatter) for quick checks
    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(feature_matrix)

    fig, ax = plt.subplots(figsize=(9, 7))

    # Use a larger qualitative colormap for more distinct cluster colors
    max_colors = 20
    if n_clusters <= max_colors:
        cmap = plt.cm.get_cmap('tab20')
        color_vals = cmap(labels % max_colors)
    else:
        # fallback: HSV palette for many clusters
        color_vals = plt.cm.hsv(labels / float(n_clusters))

    # Add small jitter to prevent overlapping points from being hidden
    jitter_scale = 0.02
    np.random.seed(42)
    features_2d_jittered = features_2d + np.random.normal(0, jitter_scale, features_2d.shape)

    scatter = ax.scatter(
        features_2d_jittered[:, 0], features_2d_jittered[:, 1],
        c=color_vals, s=150, alpha=1.0, edgecolors='black', linewidths=1.0
    )

    ax.set_xlabel('First Principal Component', fontsize=12)
    ax.set_ylabel('Second Principal Component', fontsize=12)
    ax.set_title(f'K-Means Clustering ({metric.title()}, K={n_clusters}) - PCA (PC1 vs PC2)', fontsize=14)
    ax.grid(True, alpha=0.25)

    # Create a clear legend mapping cluster id -> color
    handles = []
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        col = color_vals[labels == lab][0]
        handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {int(lab)}',
                                  markerfacecolor=col, markersize=12, markeredgecolor='black'))
    ax.legend(handles=handles, title='Clusters', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    # Print concise cluster statistics (kept from the original function)
    print("\n" + "="*60)
    print("Cluster Statistics")
    print("="*60)
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Cluster {label}: {count} students")
    print(f"Total students: {len(labels)}")
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    # If dataframe provided, print group membership details
    if df is not None and 'Name' in df.columns:
        for label in unique_labels:
            members = df.loc[labels == label, 'Name'].tolist()
            print(f"\nGroup {int(label) + 1} ({len(members)} students):")
            print(f"  Students: {', '.join(members)}")
            if 'Gender' in df.columns:
                print(f"  Gender: {df.loc[labels == label, 'Gender'].value_counts().to_dict()}")
            if 'Diversity' in df.columns:
                print(f"  Diversity: {df.loc[labels == label, 'Diversity'].value_counts().to_dict()}")


def visualize_feature_pairs(
    df: pd.DataFrame,
    labels: np.ndarray,
    n_clusters: int
):
    """
    Visualize clusters using different feature pairs.
    
    Args:
        df: DataFrame with student data
        labels: Cluster labels
        n_clusters: Number of clusters
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Feature pairs to visualize
    feature_pairs = [
        ('Motivation', 'Self_Esteem'),
        ('Motivation', 'Work_Ethic'),
        ('Self_Esteem', 'Work_Ethic') 
    ]
    
    for idx, (feat1, feat2) in enumerate(feature_pairs[:3]):
        ax = axes[idx]

        # strong contrasting colors per cluster
        if n_clusters <= 20:
            cmap_fp = plt.cm.get_cmap('tab20')
            color_vals_fp = cmap_fp(labels % 20)
        else:
            color_vals_fp = plt.cm.hsv(labels / float(n_clusters))

        # Add jitter to avoid overlapping points
        jitter_scale = 0.05
        np.random.seed(42)
        feat1_jittered = df[feat1].values + np.random.normal(0, jitter_scale, len(df))
        feat2_jittered = df[feat2].values + np.random.normal(0, jitter_scale, len(df))

        scatter = ax.scatter(
            feat1_jittered, feat2_jittered,
            c=color_vals_fp, s=140, alpha=1.0, edgecolors='black', linewidths=1.2
        )
        ax.set_xlabel(feat1, fontsize=12)
        ax.set_ylabel(feat2, fontsize=12)
        ax.set_title(f'Clusters: {feat1} vs {feat2}', fontsize=12, fontweight='bold')
        ax.set_xlim(0.5, 4.5)
        ax.set_ylim(0.5, 4.5)
        ax.grid(True, alpha=0.25)

        # add small legend for cluster ids (only on first subplot to avoid clutter)
        if idx == 0:
            handles_fp = []
            for lab in np.unique(labels):
                col = color_vals_fp[labels == lab][0]
                handles_fp.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {int(lab)}',
                                             markerfacecolor=col, markersize=10, markeredgecolor='black'))
            ax.legend(handles=handles_fp, title='Clusters', bbox_to_anchor=(1.02, 1), loc='upper left')
        else:
            # keep colorbar for context
            plt.colorbar(scatter, ax=ax, label='Cluster')
    
    # Fourth subplot: Cluster size distribution
    ax = axes[3]
    unique_labels, counts = np.unique(labels, return_counts=True)
    # Use tab20 palette for clearer, higher-contrast colors
    palette_bars = plt.cm.get_cmap('tab20')
    bar_colors = [palette_bars(int(l) % 20) for l in unique_labels]
    bars = ax.bar(unique_labels, counts, color=bar_colors, alpha=0.95, edgecolor='black', linewidth=1.2)
    ax.set_xlabel('Cluster ID', fontsize=12)
    ax.set_ylabel('Number of Students', fontsize=12)
    ax.set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
    ax.set_xticks(unique_labels)
    ax.grid(True, alpha=0.25, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle(f'K-Means Clustering Visualization (K={n_clusters})', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()


def visualize_clustering_with_centroids(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    centroids: Optional[np.ndarray] = None
):
    """
    Visualize clustering with centroids shown.
    
    Args:
        feature_matrix: NxM feature matrix
        labels: Cluster labels
        n_clusters: Number of clusters
        centroids: Optional centroids to display
    """
    # Reduce to 2D using PCA
    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(feature_matrix)
    
    # Compute centroids if not provided
    if centroids is None:
        centroids = np.array([feature_matrix[labels == k].mean(axis=0) 
                              for k in range(n_clusters)])
    
    # Transform centroids to 2D
    centroids_2d = pca.transform(centroids)
    
    # Plot
    plt.figure(figsize=(10, 8))

    # strong color mapping for clusters
    if n_clusters <= 20:
        cmap_c = plt.cm.get_cmap('tab20')
        color_vals_c = cmap_c(labels % 20)
    else:
        color_vals_c = plt.cm.hsv(labels / float(n_clusters))

    # Add jitter to PCA features to reveal overlapping points
    jitter_scale = 0.02
    np.random.seed(42)
    features_2d_jittered_c = features_2d + np.random.normal(0, jitter_scale, features_2d.shape)

    scatter = plt.scatter(features_2d_jittered_c[:, 0], features_2d_jittered_c[:, 1], c=color_vals_c,
                         s=140, alpha=1.0, edgecolors='black', linewidths=1.2)
    
    # Plot centroids with high contrast marker
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], facecolors='none', edgecolors='black', marker='X',
               s=350, linewidths=2.5, label='Centroids', zorder=5)
    
    plt.xlabel('First Principal Component', fontsize=12)
    plt.ylabel('Second Principal Component', fontsize=12)
    plt.title(f'K-Means Clustering with Centroids (K={n_clusters})', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.show()


# =============================================================================
# Fairness post-processing — only ``labels`` change; ``df`` profile data is read-only
# Swaps are 1-for-1 between two groups, so Step 6 group sizes stay the same.
# =============================================================================

from .fairness_distribution import (
    MIN_GROUP_SIZE_DIVERSITY as _MIN_GROUP_SIZE_DIVERSITY,
    MIN_GROUP_SIZE_GENDER as _MIN_GROUP_SIZE_GENDER,
    category_target_counts,
    has_fairness_donor as _has_fairness_donor,
    rebalance_demographic_column,
)


def _gender_isolation_in_group(
    df: pd.DataFrame, labels: np.ndarray, group_id: int
) -> Optional[str]:
    """First gender category with exactly one member in this group, or None."""
    group_mask = labels == group_id
    if int(group_mask.sum()) < _MIN_GROUP_SIZE_GENDER:
        return None
    counts = df.loc[group_mask, "Gender"].value_counts()
    for gender_value, count in counts.items():
        if count == 1:
            return str(gender_value)
    return None


def _diversity_isolation_in_group(
    df: pd.DataFrame, labels: np.ndarray, group_id: int
) -> Optional[str]:
    """First diversity category with exactly one member in this group, or None."""
    group_mask = labels == group_id
    if int(group_mask.sum()) < _MIN_GROUP_SIZE_DIVERSITY:
        return None
    counts = df.loc[group_mask, "Diversity"].value_counts()
    for category, count in counts.items():
        if count == 1:
            return str(category)
    return None


def _find_best_fairness_swap(
    df: pd.DataFrame,
    labels: np.ndarray,
    distance_matrix: np.ndarray,
    group_id: int,
    target_value: str,
    column: str,
) -> Optional[tuple[int, int, int]]:
    """
    Pick a 1-for-1 swap: bring ``target_value`` into ``group_id`` from a donor group.

    Prefers donors with 3+ of ``target_value`` so the donor does not become newly
    isolated after the swap; falls back to donors with exactly 2.
    """
    unique_labels = np.unique(labels)
    current_indices = np.where(labels == group_id)[0]
    candidates_out = [
        i for i in current_indices if str(df.iloc[i][column]) != target_value
    ]
    if not candidates_out:
        return None

    # Tier 1: donor still has 2+ of target after giving one away; tier 2: donor had 2.
    for min_donor_count in (3, 2):
        best_swap: Optional[tuple[int, int, int]] = None
        min_cost = float("inf")

        for donor_id in unique_labels:
            if donor_id == group_id:
                continue

            donor_indices = np.where(labels == donor_id)[0]
            donor_values = df.iloc[donor_indices][column]
            if int((donor_values == target_value).sum()) < min_donor_count:
                continue

            candidates_in = [
                i for i in donor_indices if str(df.iloc[i][column]) == target_value
            ]
            for c_in in candidates_in:
                for c_out in candidates_out:
                    dist = float(distance_matrix[c_in, c_out])
                    if dist < min_cost:
                        min_cost = dist
                        best_swap = (c_in, c_out, int(donor_id))

        if best_swap is not None:
            return best_swap

    return None


def check_gender_isolation(df: pd.DataFrame, labels: np.ndarray) -> Dict[int, str]:
    """
    Flag groups where a student is the only member of their gender (any label).

    Applies to all Gender values (Male, Female, Non-binary, etc.), not only M/F.
    Groups with fewer than 4 students are skipped (same threshold as before).
    """
    isolation_dict: Dict[int, str] = {}
    for label in np.unique(labels):
        isolated = _gender_isolation_in_group(df, labels, int(label))
        if isolated is not None:
            isolation_dict[int(label)] = isolated
    return isolation_dict


def fix_gender_isolation(
    df: pd.DataFrame,
    labels: np.ndarray,
    distance_matrix: np.ndarray,
    isolated_groups: Dict[int, str],
    *,
    verbose: bool = False,
) -> tuple[np.ndarray, bool]:
    """
    Fix gender isolation for any gender category (same swap pattern as diversity).

    Returns updated labels and whether any swap was applied.
    """
    labels = labels.copy()
    changed = False

    for group_id in sorted(isolated_groups):
        while True:
            target_gender = _gender_isolation_in_group(df, labels, group_id)
            if target_gender is None:
                break
            best_swap = _find_best_fairness_swap(
                df, labels, distance_matrix, group_id, target_gender, "Gender"
            )
            if best_swap is None:
                break
            s_in, s_out, d_id = best_swap
            labels[s_in] = group_id
            labels[s_out] = d_id
            changed = True
            if verbose:
                print(
                    f"Fixed gender ({target_gender}) in G{group_id}: "
                    f"Swapped {s_in} with {s_out}"
                )

    return labels, changed


def check_diversity_isolation(df: pd.DataFrame, labels: np.ndarray) -> Dict[int, str]:
    """
    Check if any group has diversity isolation (e.g., 1 Black student in a group of 4).
    """
    isolation_dict: Dict[int, str] = {}
    for label in np.unique(labels):
        isolated = _diversity_isolation_in_group(df, labels, int(label))
        if isolated is not None:
            isolation_dict[int(label)] = isolated
    return isolation_dict


def fix_diversity_isolation(
    df: pd.DataFrame,
    labels: np.ndarray,
    distance_matrix: np.ndarray,
    isolated_groups: Dict[int, str],
    *,
    verbose: bool = False,
) -> tuple[np.ndarray, bool]:
    """
    Fix diversity isolation by finding a donor group with EXTRA students of that category.

    Returns updated labels and whether any swap was applied.
    """
    labels = labels.copy()
    changed = False

    for group_id in sorted(isolated_groups):
        while True:
            target_category = _diversity_isolation_in_group(df, labels, group_id)
            if target_category is None:
                break
            best_swap = _find_best_fairness_swap(
                df,
                labels,
                distance_matrix,
                group_id,
                target_category,
                "Diversity",
            )
            if best_swap is None:
                break
            s_in, s_out, d_id = best_swap
            labels[s_in] = group_id
            labels[s_out] = d_id
            changed = True
            if verbose:
                print(
                    f"Fixed Diversity ({target_category}) in G{group_id}: "
                    f"Swapped {s_in} with {s_out}"
                )

    return labels, changed

def form_balanced_groups(
    df: pd.DataFrame,
    desired_group_size: int,
    random_state: Optional[int] = None,
    enforce_gender_balance: bool = True,
    enforce_diversity_balance: bool = True,
    visualize: bool = False
) -> Tuple[pd.DataFrame, Dict[int, List[str]]]:
    """
    LEGACY / RESEARCH ONLY — not used by CLI, API, or ``run_grouping_pipeline``.

    Uses K-Means (Euclidean) and Euclidean swap costs. Use ``run_grouping_pipeline`` instead.
    """
    # --- Legacy path below: classroom study should use pipeline.py instead ---
    n_students = len(df)
    n_groups = max(1, n_students // desired_group_size)
    # Step 1: Compute feature vectors
    feature_matrix = compute_feature_vector(df)
    
    # Step 2: Compute distance matrices
    # CRITICAL FIX: We must unpack the 3 return values.
    # We need 'euc_dist' for the swapping logic (to find closest match)
    # We need 'man_dist' implicitly if we were running K-Medoids here, 
    # but initial_clustering usually handles its own distance calc unless passed.
    euc_dist, _, _ = compute_distance_matrix(df, feature_matrix)
    
    # Step 3: Initial clustering using K-Means
    # (Note: If you want K-Medoids here, you should swap this call, 
    # but for now we stick to the default initial_clustering function)
    labels = initial_clustering(feature_matrix, n_groups, random_state, visualize=visualize, df=df)

    labels = enforce_group_size(
        labels,
        desired_group_size,
        feature_matrix=feature_matrix,
    )
    
    if enforce_gender_balance:
        labels, _ = rebalance_demographic_column(
            df, labels, euc_dist, "Gender", min_group_size=_MIN_GROUP_SIZE_GENDER
        )
    if enforce_diversity_balance:
        labels, _ = rebalance_demographic_column(
            df,
            labels,
            euc_dist,
            "Diversity",
            min_group_size=_MIN_GROUP_SIZE_DIVERSITY,
        )
    
    # Step 6: Add Group column to dataframe (1-indexed)
    result_df = df.copy()
    result_df['Group'] = labels + 1
    
    # Step 7: Create groups dictionary
    groups_dict = {}
    for group_id in range(1, n_groups + 1):
        group_students = result_df[result_df['Group'] == group_id]['Name'].tolist()
        groups_dict[group_id] = group_students
    
    return result_df, groups_dict
