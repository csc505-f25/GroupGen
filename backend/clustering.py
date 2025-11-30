
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
import gower
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from .kmedoids import kmedoids_pam


def enforce_group_size(
    labels: np.ndarray, 
    group_size: int, 
    feature_matrix: np.ndarray = None,
    metric: str = 'manhattan'  # <--- NEW: Match your clustering metric
):
    """
    Redistributes students to enforce target group size.
    Places remainder students into the group they are closest to
    using the specified distance metric (default: manhattan).
    """
    n_students = len(labels)
    n_groups = n_students // group_size
    
    if n_groups == 0:
        return np.zeros(n_students, dtype=int)
    
    remainder = n_students % group_size
    
    # Sort to keep core clusters together
    sorted_indices = np.argsort(labels)
    new_labels = np.full(n_students, -1, dtype=int)
    
    # Assign the "Core" (Perfect groups of 5)
    cutoff = n_groups * group_size
    core_indices = sorted_indices[:cutoff]
    remainder_indices = sorted_indices[cutoff:]
    
    for i, idx in enumerate(core_indices):
        new_labels[idx] = i // group_size
        
    # Assign the "Remainder" (The 31st student)
    if remainder > 0:
        if feature_matrix is not None:
            # 1. Calculate the center (mean) of the new core groups
            group_centers = []
            for g_id in range(n_groups):
                members = np.where(new_labels == g_id)[0]
                group_centers.append(feature_matrix[members].mean(axis=0))
            
            group_centers = np.array(group_centers)
            
            # 2. Assign leftover students to the closest group center
            # using the specific metric (Manhattan)
            for idx in remainder_indices:
                student_features = feature_matrix[idx].reshape(1, -1)
                
                # Compute distance to all group centers
                dists = pairwise_distances(student_features, group_centers, metric=metric)
                
                # Find the index of the minimum distance
                closest_group = np.argmin(dists)
                new_labels[idx] = closest_group
                
        else:
            # Fallback if no features provided
            for i, idx in enumerate(remainder_indices):
                new_labels[idx] = i % n_groups

    return new_labels
    

def compute_feature_vector(df):
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


def compute_distance_matrix(df, feature_matrix):
    """
    Compute pairwise distance matrix between students.
    
    Args:
        feature_matrix: NxM array of features
        
    Returns:
        NxN distance matrix (Euclidean distance)
    """
    # Compute pairwise distances
    eucdistance_matrix = pairwise_distances(feature_matrix, metric='euclidean')
    mandistance_matrix = pairwise_distances(feature_matrix, metric='manhattan')

    # Identify the columns needed for Gower
    gower_cols = ['Motivation', 'Self_Esteem', 'Work_Ethic', 'Gender', 'Diversity', 'Learning_Style']
    gower_data = df[gower_cols].copy()

    #Which columns are categorical
    categorical_cols = [gower_data[col].dtype == 'object' for col in gower_cols]

    # Compute Gower distance matrix
    gower_distance_matrix = gower.gower_matrix(gower_data, cat_features=categorical_cols)

    return eucdistance_matrix, mandistance_matrix, gower_distance_matrix


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
    n_clusters: int = 3, 
    random_state: Optional[int] = 42, 
    group_size: int = None, 
    use_kmedoids: bool = True):
    """
    Compute feature vectors from `df` and run clustering with both Euclidean and
    Manhattan assignment metrics, visualizing each result using existing helpers.

    This function is a convenience wrapper to help you compare how the two
    metrics behave on the same data. It does not modify `df`.

    Args:
        df: Input DataFrame with student data
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility
        group_size: Optional; if set, enforce balanced group sizes via post-processing
        use_kmedoids: If True, use K-Medoids (PAM) for manhattan metric; if False, use K-Means for both
    """
    feature_matrix = compute_feature_vector(df)

    for metric in ("euclidean", "manhattan"):
        print(f"\n=== Running clustering with metric: {metric} (K={n_clusters}) ===")

        # Choose algorithm: K-Medoids for Manhattan (if use_kmedoids=True), K-Means for Euclidean
        if metric == "manhattan" and use_kmedoids:
            labels, medoid_indices = kmedoids_pam(feature_matrix, n_clusters, distance_metric=metric, random_state=random_state)
            centroids = feature_matrix[medoid_indices]  # For visualization
            print(f"Using K-Medoids (medoid indices: {medoid_indices})")
        else:
            labels, centroids = kmeans_custom(feature_matrix, n_clusters, random_state=random_state, return_centroids=True, metric=metric)
            print(f"Using K-Means")

        if group_size is not None:
            labels = enforce_group_size(labels, group_size)
            print(f"Groups forcibly balanced to {group_size} students each.")
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
        ('Self_Esteem', 'Work_Ethic') # Will be replaced with 3D or different view
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


def check_gender_isolation(pd, labels):
    """
    Check if any group has gender isolation (e.g., all men and one woman).
    
    Args:
        df: DataFrame with student data (must have 'Gender' column)
        labels: Cluster labels for each student
        
    Returns:
        Dictionary mapping group_id to isolation_type
        Example: {2: 'female_isolated'} means group 2 has one female isolated
    """
    isolation_dict = {}
    unique_labels = np.unique(labels)

    for label in unique_labels:
        #get only students in this specific cluster
        group_mask = (labels == label)
        group_size = group_mask.sum()

        #if group too small, skip
        if group_size < 4:
            continue
        
        #fetch the data for this group
        group_genders = pd.loc[group_mask, 'Gender']
        #Count number of males and females
        n_males = (group_genders == 'Male').sum()
        n_females = (group_genders == 'Female').sum()
        
        #isolation logic
        if n_females == 1 and n_males > 1:
            isolation_dict[int(label)] = 'female_isolated'
            
        # Condition B: Male Isolated (1 Male, > 1 Females)
        elif n_males == 1 and n_females > 1:
            isolation_dict[int(label)] = 'male_isolated'
            
    return isolation_dict


# def check_diversity_isolation(df: pd.DataFrame, labels: np.ndarray) -> Dict[int, str]:
#     """
#     Check if any group has diversity isolation (e.g., all white and one black).
    
#     Args:
#         df: DataFrame with student data (must have 'Diversity' column)
#         labels: Cluster labels for each student
        
#     Returns:
#         Dictionary mapping group_id to isolation_type
#         Example: {3: 'black_isolated'} means group 3 has one black student isolated
        
#     TODO: Implement diversity isolation detection
#     - For each group, count diversity categories
#     - If a group has only 1 student of a particular diversity category and others are different,
#       mark that category as isolated
#     - Return dictionary of isolated groups with their isolated category
#     """
#     # TODO: For each unique cluster label:
#     #   - Count diversity categories in that group
#     #   - For each diversity category, if count == 1 and total group size > 1:
#     #     - Mark as '{category}_isolated'
#     # TODO: Return dictionary of isolated groups
#     pass


def fix_gender_isolation(
    df: pd.DataFrame,
    labels: np.ndarray,
    distance_matrix: np.ndarray,
    isolated_groups: Dict[int, str]
) -> np.ndarray:
    """
    Fix gender isolation by swapping students to ensuring no one is the 'only one' 
    of their gender in a group > 3.
    """
    labels = labels.copy() # Work on a copy to avoid accidental side effects
    
    # Iterate through each group that was flagged as isolated
    for group_id, isolation_type in isolated_groups.items():
        
        # 1. SETUP: Determine what we need to find and what we need to give away
        # ---------------------------------------------------------------------
        target_gender = None   # The gender we need to BRING IN
        swap_out_gender = None # The gender we need to SEND OUT
        
        if isolation_type == 'female_isolated':
            # Group has 1 Female, many Males. 
            # We need another Female. We will trade away a Male.
            target_gender = 'Female'
            swap_out_gender = 'Male'
        elif isolation_type == 'male_isolated':
            # Group has 1 Male, many Females.
            # We need another Male. We will trade away a Female.
            target_gender = 'Male'
            swap_out_gender = 'Female'
            
        # Get indices of students currently in this isolated group
        current_group_indices = np.where(labels == group_id)[0]
        
        # Identify the candidates IN THIS GROUP who can be swapped out
        # (e.g., if we need a Female, we look for a Male to swap out)
        candidates_to_swap_out = [
            idx for idx in current_group_indices 
            if df.iloc[idx]['Gender'] == swap_out_gender
        ]
        
        if not candidates_to_swap_out:
            continue # Should not happen if check_gender_isolation is correct, but safety first

        # 2. FIND A DONOR: Look for the best swap from other groups
        # ---------------------------------------------------------------------
        best_swap = None
        min_cost = float('inf')
        
        unique_labels = np.unique(labels)
        
        for donor_group_id in unique_labels:
            if donor_group_id == group_id:
                continue # Cannot swap with self
            
            # Get donor group members
            donor_indices = np.where(labels == donor_group_id)[0]
            donor_genders = df.iloc[donor_indices]['Gender']
            
            # CHECK: Does this donor group have enough of the target gender?
            # We generally only want to take from a group if they have > 2 of that gender,
            # so we don't accidentally create a NEW isolation problem there.
            count_target = (donor_genders == target_gender).sum()
            
            if count_target > 2: # Safe to take one
                
                # Identify candidates in the DONOR group who are the target gender
                candidates_to_bring_in = [
                    idx for idx in donor_indices 
                    if df.iloc[idx]['Gender'] == target_gender
                ]
                
                # 3. CALCULATE COST: Find the pair with minimum distance
                # -------------------------------------------------------------
                # We want the student coming IN to be similar to the students currently 
                # in the isolated group.
                
                for candidate_in in candidates_to_bring_in:
                    # Calculate average distance from this candidate to the REST of the isolated group
                    # (excluding the person we are swapping out)
                    
                    # Simple heuristic: Just minimize the distance between the IN and OUT students
                    # (This is often sufficient and faster than full group centroid recalc)
                    for candidate_out in candidates_to_swap_out:
                        
                        # Distance between the student leaving and student entering
                        # Lower distance = they are similar = less disruption to group chemistry
                        dist = distance_matrix[candidate_in, candidate_out]
                        
                        if dist < min_cost:
                            min_cost = dist
                            best_swap = (candidate_in, candidate_out, donor_group_id)

        # 4. EXECUTE SWAP
        # ---------------------------------------------------------------------
        if best_swap:
            student_in, student_out, donor_id = best_swap
            
            # Perform the swap in the labels array
            labels[student_in] = group_id   # Move donor student to isolated group
            labels[student_out] = donor_id  # Move isolated group student to donor group
            
            print(f"Fixed {isolation_type} in Group {group_id}: Swapped {student_in} (from G{donor_id}) with {student_out}")
            
    return labels


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
