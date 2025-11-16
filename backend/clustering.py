"""
Clustering Module

This module implements clustering algorithms to group students.
You need to implement:
1. Feature-based similarity/distance computation
2. K-Medoids or similar clustering algorithm
3. Gender and diversity locking mechanism to prevent isolation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder


def compute_feature_vector(df: pd.DataFrame) -> np.ndarray:
    """
    Convert student features to numerical vectors for clustering.
    
    Features to include:
    - Motivation (1-4)
    - Self_Esteem (1-4)
    - Work_Ethic (1-4)
    - Learning_Style (encode as numeric, e.g., one-hot or label encoding)
    
    Args:
        df: DataFrame with student data
        
    Returns:
        NxM numpy array where N is number of students, M is number of features
    """
    # Extract numeric features
    numeric_features = df[['Motivation', 'Self_Esteem', 'Work_Ethic']].values
    
    # Encode Learning_Style using OneHotEncoder
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    learning_style_encoded = one_hot_encoder.fit_transform(df[['Learning_Style']])
    
    # Combine features
    features = np.hstack([numeric_features, learning_style_encoded])
    
    # Normalize features using StandardScaler for better clustering
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    return features


def compute_distance_matrix(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distance matrix between students.
    
    Args:
        feature_matrix: NxM array of features
        
    Returns:
        NxN distance matrix (Euclidean distance)
    """
    # Compute pairwise Euclidean distances
    distance_matrix = pairwise_distances(feature_matrix, metric='euclidean')
    distance_matrix2 = pairwise_distances(feature_matrix, metric='manhattan')
    return distance_matrix, distance_matrix2


def kmeans_custom(x: np.ndarray, K: int, random_state: Optional[int] = None, max_iter: int = 300, return_centroids: bool = False, metric: str = 'euclidean'):
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
) -> np.ndarray:
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


def compare_metrics_and_visualize(df: pd.DataFrame, n_clusters: int = 3, random_state: Optional[int] = 42) -> None:
    """
    Compute feature vectors from `df` and run clustering with both Euclidean and
    Manhattan assignment metrics, visualizing each result using existing helpers.

    This function is a convenience wrapper to help you compare how the two
    metrics behave on the same data. It does not modify `df`.
    """
    feature_matrix = compute_feature_vector(df)

    for metric in ("euclidean", "manhattan"):
        print(f"\n=== Running clustering with metric: {metric} (K={n_clusters}) ===")
        labels, centroids = kmeans_custom(feature_matrix, n_clusters, random_state=random_state, return_centroids=True, metric=metric)
        visualize_clustering(feature_matrix, labels, n_clusters, df=df)


def visualize_clustering(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    df: Optional[pd.DataFrame] = None
):
    """
    Visualize clustering results using PCA to reduce dimensions to 2D.
    
    Args:
        feature_matrix: NxM feature matrix
        labels: Cluster labels for each student
        n_clusters: Number of clusters
        df: Optional DataFrame with student data (for displaying names)
    """
    # Reduce dimensions to 2D using PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(feature_matrix)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Clusters in 2D space (PCA)
    ax1 = axes[0]
    scatter = ax1.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, 
                         cmap='tab10', s=100, alpha=0.6, edgecolors='black', linewidths=1)
    ax1.set_xlabel('First Principal Component', fontsize=12)
    ax1.set_ylabel('Second Principal Component', fontsize=12)
    ax1.set_title(f'K-Means Clustering (K={n_clusters}) - PCA Visualization', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Cluster')
    
    # Plot 2: Feature pairs visualization (Motivation vs Self_Esteem)
    ax2 = axes[1]
    if df is not None and 'Motivation' in df.columns and 'Self_Esteem' in df.columns:
        scatter2 = ax2.scatter(df['Motivation'], df['Self_Esteem'], c=labels,
                              cmap='tab10', s=100, alpha=0.6, edgecolors='black', linewidths=1)
        ax2.set_xlabel('Motivation (1-4)', fontsize=12)
        ax2.set_ylabel('Self-Esteem (1-4)', fontsize=12)
        ax2.set_title('Clusters: Motivation vs Self-Esteem', fontsize=14, fontweight='bold')
        ax2.set_xlim(0.5, 4.5)
        ax2.set_ylim(0.5, 4.5)
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Cluster')
    else:
        # If no dataframe provided, show another PCA view
        scatter2 = ax2.scatter(features_2d[:, 0], features_2d[:, 1], c=labels,
                              cmap='tab10', s=100, alpha=0.6, edgecolors='black', linewidths=1)
        ax2.set_xlabel('First Principal Component', fontsize=12)
        ax2.set_ylabel('Second Principal Component', fontsize=12)
        ax2.set_title('Clustering Results (Alternative View)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Cluster')
    
    plt.tight_layout()
    plt.show()
    
    # Print cluster statistics
    print("\n" + "="*60)
    print("Cluster Statistics")
    print("="*60)
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Cluster {label}: {count} students")
    print(f"Total students: {len(labels)}")
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.2%}")


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
        ('Self_Esteem', 'Work_Ethic'),
        ('Motivation', 'Self_Esteem')  # Will be replaced with 3D or different view
    ]
    
    for idx, (feat1, feat2) in enumerate(feature_pairs[:3]):
        ax = axes[idx]
        scatter = ax.scatter(df[feat1], df[feat2], c=labels,
                            cmap='tab10', s=100, alpha=0.6, edgecolors='black', linewidths=1)
        ax.set_xlabel(feat1, fontsize=12)
        ax.set_ylabel(feat2, fontsize=12)
        ax.set_title(f'Clusters: {feat1} vs {feat2}', fontsize=12, fontweight='bold')
        ax.set_xlim(0.5, 4.5)
        ax.set_ylim(0.5, 4.5)
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Cluster')
    
    # Fourth subplot: Cluster size distribution
    ax = axes[3]
    unique_labels, counts = np.unique(labels, return_counts=True)
    bars = ax.bar(unique_labels, counts, color=plt.cm.tab10(unique_labels / n_clusters), 
                  alpha=0.7, edgecolor='black')
    ax.set_xlabel('Cluster ID', fontsize=12)
    ax.set_ylabel('Number of Students', fontsize=12)
    ax.set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
    ax.set_xticks(unique_labels)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
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
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels,
                         cmap='tab10', s=100, alpha=0.6, edgecolors='black', linewidths=1)
    
    # Plot centroids
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='X',
               s=300, edgecolors='black', linewidths=2, label='Centroids', zorder=5)
    
    plt.xlabel('First Principal Component', fontsize=12)
    plt.ylabel('Second Principal Component', fontsize=12)
    plt.title(f'K-Means Clustering with Centroids (K={n_clusters})', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.show()


def check_gender_isolation(df: pd.DataFrame, labels: np.ndarray) -> Dict[int, str]:
    """
    Check if any group has gender isolation (e.g., all men and one woman).
    
    Args:
        df: DataFrame with student data (must have 'Gender' column)
        labels: Cluster labels for each student
        
    Returns:
        Dictionary mapping group_id to isolation_type
        Example: {2: 'female_isolated'} means group 2 has one female isolated
        
    TODO: Implement gender isolation detection
    - For each group, count males and females
    - If a group has only 1 female and rest are male, mark as 'female_isolated'
    - If a group has only 1 male and rest are female, mark as 'male_isolated'
    - Return dictionary of isolated groups
    """
    # TODO: For each unique cluster label:
    #   - Count number of males and females in that group
    #   - If (females == 1 and males > 1), mark as 'female_isolated'
    #   - If (males == 1 and females > 1), mark as 'male_isolated'
    # TODO: Return dictionary of isolated groups
    pass


def check_diversity_isolation(df: pd.DataFrame, labels: np.ndarray) -> Dict[int, str]:
    """
    Check if any group has diversity isolation (e.g., all white and one black).
    
    Args:
        df: DataFrame with student data (must have 'Diversity' column)
        labels: Cluster labels for each student
        
    Returns:
        Dictionary mapping group_id to isolation_type
        Example: {3: 'black_isolated'} means group 3 has one black student isolated
        
    TODO: Implement diversity isolation detection
    - For each group, count diversity categories
    - If a group has only 1 student of a particular diversity category and others are different,
      mark that category as isolated
    - Return dictionary of isolated groups with their isolated category
    """
    # TODO: For each unique cluster label:
    #   - Count diversity categories in that group
    #   - For each diversity category, if count == 1 and total group size > 1:
    #     - Mark as '{category}_isolated'
    # TODO: Return dictionary of isolated groups
    pass


def fix_gender_isolation(
    df: pd.DataFrame,
    labels: np.ndarray,
    distance_matrix: np.ndarray,
    isolated_groups: Dict[int, str]
) -> np.ndarray:
    """
    Fix gender isolation by moving students to ensure no one is alone.
    
    Strategy:
    - If a group has one female isolated among males, find another female from
      another group and swap with a male from that group
    - Similar for male isolation
    
    Args:
        df: DataFrame with student data
        labels: Current cluster labels
        distance_matrix: Distance matrix for finding good swaps
        isolated_groups: Dictionary of isolated groups from check_gender_isolation()
        
    Returns:
        Updated cluster labels
        
    TODO: Implement gender isolation fixing
    - For each isolated group:
    #   - If female_isolated: find another group with a female, swap her with a male from isolated group
    #   - If male_isolated: find another group with a male, swap him with a female from isolated group
    #   - Try to minimize distance increase when swapping (find closest match)
    # TODO: Return updated labels
    """
    # TODO: For each isolated group:
    #   - Identify the isolated student and their gender
    #   - Find another student of same gender from a different group
    #   - Swap them (or move the other student to isolated group)
    #   - Consider distance when choosing which student to swap (minimize distance increase)
    # TODO: Return updated labels
    pass


def fix_diversity_isolation(
    df: pd.DataFrame,
    labels: np.ndarray,
    distance_matrix: np.ndarray,
    isolated_groups: Dict[int, str]
) -> np.ndarray:
    """
    Fix diversity isolation by moving students to ensure no one is alone.
    
    Strategy:
    - If a group has one student of a diversity category isolated, find another
      student of the same category from another group and add them to the isolated group
    - Or swap with a student from the isolated group
    
    Args:
        df: DataFrame with student data
        labels: Current cluster labels
        distance_matrix: Distance matrix for finding good swaps
        isolated_groups: Dictionary of isolated groups from check_diversity_isolation()
        
    Returns:
        Updated cluster labels
        
    TODO: Implement diversity isolation fixing
    - For each isolated group:
    #   - Identify the isolated student and their diversity category
    #   - Find another student of same diversity category from a different group
    #   - Move that student to the isolated group (or swap)
    #   - Consider distance when choosing which student to move
    # TODO: Return updated labels
    """
    # TODO: For each isolated group:
    #   - Identify the isolated student and their diversity category
    #   - Find another student of same diversity category from a different group
    #   - Move them to the isolated group (try to minimize distance increase)
    # TODO: Return updated labels
    pass


def form_balanced_groups(
    df: pd.DataFrame,
    n_groups: int,
    random_state: Optional[int] = None,
    enforce_gender_balance: bool = True,
    enforce_diversity_balance: bool = True,
    visualize: bool = False
) -> Tuple[pd.DataFrame, Dict[int, List[str]]]:
    """
    Main function to form balanced student groups with locking mechanisms.
    
    Steps:
    1. Compute feature vectors and distance matrix
    2. Perform initial clustering
    3. Check for gender isolation and fix if needed
    4. Check for diversity isolation and fix if needed
    5. Return final group assignments
    
    Args:
        df: DataFrame with student data
        n_groups: Number of groups to form
        random_state: Random seed
        enforce_gender_balance: Whether to enforce gender balance
        enforce_diversity_balance: Whether to enforce diversity balance
        visualize: Whether to visualize clustering results
        
    Returns:
        Tuple of (DataFrame with Group column added, Dictionary mapping group_id to student names)
        
    TODO: Implement the complete workflow
    - Call compute_feature_vector() to get features
    - Call compute_distance_matrix() to get distances
    - Call initial_clustering() to get initial groups
    - If enforce_gender_balance: check and fix gender isolation
    - If enforce_diversity_balance: check and fix diversity isolation
    - Add 'Group' column to dataframe (groups numbered 1, 2, 3, ...)
    - Create dictionary mapping group_id to list of student names
    - Return results
    """
    # Step 1: Compute feature vectors
    feature_matrix = compute_feature_vector(df)
    
    # Step 2: Compute distance matrix (needed for isolation fixing)
    distance_matrix = compute_distance_matrix(feature_matrix)
    
    # Step 3: Initial clustering using K-Means
    labels = initial_clustering(feature_matrix, n_groups, random_state, visualize=visualize, df=df)
    
    # # Step 4: Check and fix gender isolation
    # if enforce_gender_balance:
    #     isolated_groups = check_gender_isolation(df, labels)
    #     if isolated_groups:
    #         labels = fix_gender_isolation(df, labels, distance_matrix, isolated_groups)
    
    # # Step 5: Check and fix diversity isolation
    # if enforce_diversity_balance:
    #     isolated_groups = check_diversity_isolation(df, labels)
    #     if isolated_groups:
    #         labels = fix_diversity_isolation(df, labels, distance_matrix, isolated_groups)
    
    # Step 6: Add Group column to dataframe (labels + 1 to make groups 1-indexed)
    result_df = df.copy()
    result_df['Group'] = labels + 1
    
    # Step 7: Create groups dictionary
    groups_dict = {}
    for group_id in range(1, n_groups + 1):
        group_students = result_df[result_df['Group'] == group_id]['Name'].tolist()
        groups_dict[group_id] = group_students
    
    return result_df, groups_dict
