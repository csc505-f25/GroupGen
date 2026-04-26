
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
import os
import pickle
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import pickle
from .kmedoids import kmedoids_pam

# ==========================================
# 1. Feature Engineering & Distances
# ==========================================

def enforce_group_size(
    labels: np.ndarray, 
    group_size: int, 
    feature_matrix: np.ndarray = None,
    metric: str = 'manhattan',
    expected_n_clusters: int = None # <--- NEW: Explicitly pass expected number of groups
):
    """
    Redistributes students to enforce target group size.
    Places remainder students into the group they are closest to
    using the specified distance metric (default: manhattan).
    """
    if len(labels) == 0:
        return labels

    n_students = len(labels)
    unique_groups_observed = np.unique(labels)
    
    # DETERMINE TOTAL GROUPS
    if expected_n_clusters is not None:
        n_groups = expected_n_clusters
        # Ensure we consider ALL groups 0..n_groups-1, even if empty
        all_groups = np.arange(n_groups)
    else:
        n_groups = len(unique_groups_observed)
        all_groups = unique_groups_observed
    
    # Copy labels to avoid modifying original array
    new_labels = labels.copy()

    # 1. Determine Target Sizes (Balanced)
    # We distribute the remainder to the first k groups
    base_size = n_students // n_groups
    remainder = n_students % n_groups
    
    # Calculate current counts (initialize 0 for all expected groups)
    current_counts = {g: 0 for g in all_groups}
    for g in new_labels:
        if g in current_counts:
            current_counts[g] += 1
    
    # Assign target capacities
    # Optimization: Assign larger targets to currently larger groups to minimize moves
    sorted_groups_by_curr_size = sorted(all_groups, key=lambda g: current_counts[g], reverse=True)
    target_sizes = {}
    for i, g_id in enumerate(sorted_groups_by_curr_size):
        target_sizes[g_id] = base_size + (1 if i < remainder else 0)

    # 2. Redistribution Loop
    max_iter = n_students * 2  # Safety limit
    
    for _ in range(max_iter):
        # Identify Donors (Have too many) and Receivers (Have too few)
        donors = [g for g in all_groups if current_counts[g] > target_sizes[g]]
        receivers = [g for g in all_groups if current_counts[g] < target_sizes[g]]
        
        if not donors or not receivers:
            break # Perfectly balanced
            
        best_move = None
        min_cost = float('inf')
        
        # If we have features, use distance. Else simplistic move.
        if feature_matrix is not None:
             # Precompute receiver centers to save time
            receiver_centers = {}
            for r_id in receivers:
                mask = (new_labels == r_id)
                if np.any(mask):
                    receiver_centers[r_id] = feature_matrix[mask].mean(axis=0)
                else:
                    # If empty, center is 0 (should rarely happen in this flow)
                    receiver_centers[r_id] = np.zeros(feature_matrix.shape[1])
            
            # Find the globally best single move from ANY donor to ANY receiver
            for d_id in donors:
                donor_indices = np.where(new_labels == d_id)[0]
                donor_features = feature_matrix[donor_indices]
                
                for r_id in receivers:
                    center_r = receiver_centers[r_id].reshape(1, -1)
                    
                    # Calculate distances from all potential donors to this receiver center
                    # We use the metric passed in (e.g., 'manhattan')
                    dists = pairwise_distances(donor_features, center_r, metric=metric).flatten()
                    
                    # Find closest student
                    local_min_idx = np.argmin(dists)
                    local_min_dist = dists[local_min_idx]
                    
                    if local_min_dist < min_cost:
                        min_cost = local_min_dist
                        best_move = (donor_indices[local_min_idx], r_id)
        
        else:
            # Fallback (No features): Just move first available student
            d_id = donors[0]
            r_id = receivers[0]
            student_idx = np.where(new_labels == d_id)[0][0]
            best_move = (student_idx, r_id)
            
        # Execute the move
        if best_move:
            student_to_move, new_group = best_move
            old_group = new_labels[student_to_move]
            
            new_labels[student_to_move] = new_group
            current_counts[old_group] -= 1
            current_counts[new_group] += 1
            
    return new_labels
    

def compute_feature_vector(df, save_scaler_path=None, load_scaler_path=None):
    """
    Convert student features to numerical vectors for clustering.
    
    Features to include:
    - Motivation (1-4)
    - Self_Esteem (1-4)
    - Work_Ethic (1-4)
    - Learning_Style (encode as numeric, e.g., one-hot or label encoding)
    
    Args:
        df: DataFrame with student data
        save_scaler_path: Optional path to save the fitted scaler
        load_scaler_path: Optional path to load a pre-fitted scaler
        
    Returns:
        NxM numpy array where N is number of students, M is number of features
    """
    # Extract numeric features
    numeric_features = df[['Motivation', 'Self_Esteem', 'Work_Ethic']].values
    
    # Encode Learning_Style using OneHotEncoder with fixed categories
    # This prevents dimension mismatch crashes on small classrooms that happen to miss a learning style
    one_hot_encoder = OneHotEncoder(categories=[['Visual', 'Auditory', 'Kinesthetic']], sparse_output=False, handle_unknown='ignore')
    learning_style_encoded = one_hot_encoder.fit_transform(df[['Learning_Style']])
    
    # Combine features
    features = np.hstack([numeric_features, learning_style_encoded])
    
    # Normalize features using StandardScaler for better clustering
    if load_scaler_path and os.path.exists(load_scaler_path):
        with open(load_scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        features = scaler.transform(features)
    else:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        if save_scaler_path:
            with open(save_scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
                
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

# ==========================================
# 2. Clustering Algorithms
# ==========================================




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
            labels, medoid_indices = kmedoids_pam(feature_matrix, n_clusters, distance_metric=metric, random_state=random_state)
            print(f"Using K-Medoids")
        else:
            labels, centroids = kmeans_custom(feature_matrix, n_clusters, random_state=random_state, return_centroids=True, metric=metric)
            print(f"Using K-Means")

        # Automatically enforce size as a secondary step
        labels = enforce_group_size(labels, desired_group_size, feature_matrix=feature_matrix, expected_n_clusters=n_clusters)
        
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
    # Save the plot explicitly using the metric name, replacing spaces with underscores
    import os
    safe_metric = metric.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')
    save_dir = os.path.join("backend", "output_plots")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{safe_metric}.png")
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")
    plt.close() # Close to prevent showing and hanging the python script

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


def check_gender_isolation(pd, labels):
    """
    Check if any group has gender isolation (e.g., all men and one woman).
    
    Args:
        pd: DataFrame with student data (must have 'Gender' column)
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
    
    for group_id, isolation_type in isolated_groups.items():
        
        # --- CRITICAL FIX 1: Re-verify the group is STILL isolated ---
        # (If we combined two isolated students earlier in this loop, 
        # this group might have already been fixed!)
        current_group_indices = np.where(labels == group_id)[0]
        current_genders = df.iloc[current_group_indices]['Gender']
        
        target_gender = 'Female' if isolation_type == 'female_isolated' else 'Male'
        swap_out_gender = 'Male' if isolation_type == 'female_isolated' else 'Female'
        
        if (current_genders == target_gender).sum() != 1:
            continue # Already fixed by a previous swap, skip!

        candidates_to_swap_out = [
            idx for idx in current_group_indices 
            if df.iloc[idx]['Gender'] == swap_out_gender
        ]
        
        if not candidates_to_swap_out: continue 

        best_swap = None
        min_cost = float('inf')
        unique_labels = np.unique(labels)
        
        for donor_group_id in unique_labels:
            if donor_group_id == group_id: continue 
            
            donor_indices = np.where(labels == donor_group_id)[0]
            donor_genders = df.iloc[donor_indices]['Gender']
            count_target = (donor_genders == target_gender).sum()
            
            # --- CRITICAL FIX 2: The "Combining" Rule ---
            # Safe if they have > 2 (leaves them with 2+)
            # OR Safe if they have exactly 1 (leaves them with 0, fixing isolation there too!)
            if count_target > 2 or count_target == 1: 
                
                candidates_to_bring_in = [
                    idx for idx in donor_indices 
                    if df.iloc[idx]['Gender'] == target_gender
                ]
                
                for candidate_in in candidates_to_bring_in:
                    for candidate_out in candidates_to_swap_out:
                        dist = distance_matrix[candidate_in, candidate_out]
                        if dist < min_cost:
                            min_cost = dist
                            best_swap = (candidate_in, candidate_out, donor_group_id)

        if best_swap:
            student_in, student_out, donor_id = best_swap
            labels[student_in] = group_id   
            labels[student_out] = donor_id  
            print(f"Fixed {isolation_type} in Group {group_id}: Swapped {student_in} (from G{donor_id}) with {student_out}")
            
    return labels


def check_diversity_isolation(df: pd.DataFrame, labels: np.ndarray) -> Dict[int, str]:
    """
    Check if any group has diversity isolation (e.g., 1 Black student in a group of 4).
    """
    isolation_dict = {}
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        group_mask = (labels == label)
        if group_mask.sum() < 3: continue
            
        group_diversity = df.loc[group_mask, 'Diversity']
        counts = group_diversity.value_counts()
        
        # If a category appears exactly ONCE, flag it
        for category, count in counts.items():
            if count == 1:
                isolation_dict[int(label)] = category
                break # Handle one isolation per group at a time
                
    return isolation_dict


def fix_diversity_isolation(df: pd.DataFrame, labels: np.ndarray, distance_matrix: np.ndarray, isolated_groups: Dict[int, str]) -> np.ndarray:
    """
    Fix diversity isolation by finding a donor group with EXTRA students of that category.
    """
    labels = labels.copy()
    unique_labels = np.unique(labels)
    
    for group_id, target_category in isolated_groups.items():
        
        current_indices = np.where(labels == group_id)[0]
        # We need to swap OUT someone who is NOT the target category
        candidates_out = [i for i in current_indices if df.iloc[i]['Diversity'] != target_category]
        
        if not candidates_out: continue
            
        best_swap = None
        min_cost = float('inf')
        
        for donor_id in unique_labels:
            if donor_id == group_id: continue
            
            donor_indices = np.where(labels == donor_id)[0]
            donor_diversity = df.iloc[donor_indices]['Diversity']
            
            # Donor must have >1 of this category. 
            # (If they have 2, and we take 1, they have 1 left. This is a trade-off. 
            # ideally >2, but for diversity categories, >1 is often the best we can find).
            if (donor_diversity == target_category).sum() > 1:
                
                candidates_in = [i for i in donor_indices if df.iloc[i]['Diversity'] == target_category]
                
                for c_in in candidates_in:
                    for c_out in candidates_out:
                        dist = distance_matrix[c_in, c_out]
                        if dist < min_cost:
                            min_cost = dist
                            best_swap = (c_in, c_out, donor_id)
        
        if best_swap:
            s_in, s_out, d_id = best_swap
            labels[s_in] = group_id
            labels[s_out] = d_id
            print(f"Fixed Diversity ({target_category}) in G{group_id}: Swapped {s_in} with {s_out}")
            
    return labels

def form_balanced_groups(
    df: pd.DataFrame,
    desired_group_size: int,
    random_state: Optional[int] = None,
    enforce_gender_balance: bool = True,
    enforce_diversity_balance: bool = True,
    visualize: bool = False
) -> Tuple[pd.DataFrame, Dict[int, List[str]]]:
    """
    Main function to form balanced student groups with locking mechanisms.
    """
    # --- DYNAMIC K CALCULATION ---
    n_students = len(df)
    n_groups = max(1, n_students // desired_group_size) # The "Table Anchor" logic

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
        expected_n_clusters=n_groups
    )
    
    # Step 4: Check and fix gender isolation
    if enforce_gender_balance:
        isolated_groups = check_gender_isolation(df, labels)
        if isolated_groups:
            print(f"   > Found gender isolation in groups: {list(isolated_groups.keys())}. Fixing...")
            # We pass 'euc_dist' here because the swap logic needs to know who is closest
            labels = fix_gender_isolation(df, labels, euc_dist, isolated_groups)
    
    # Step 5: Check and fix diversity isolation
    if enforce_diversity_balance:
        isolated_groups = check_diversity_isolation(df, labels)
        if isolated_groups:
            print(f"   > Found diversity isolation in groups: {list(isolated_groups.keys())}. Fixing...")
            labels = fix_diversity_isolation(df, labels, euc_dist, isolated_groups)
    
    # Step 6: Add Group column to dataframe (1-indexed)
    result_df = df.copy()
    result_df['Group'] = labels + 1
    
    # Step 7: Create groups dictionary
    groups_dict = {}
    for group_id in range(1, n_groups + 1):
        group_students = result_df[result_df['Group'] == group_id]['Name'].tolist()
        groups_dict[group_id] = group_students
    
    return result_df, groups_dict
