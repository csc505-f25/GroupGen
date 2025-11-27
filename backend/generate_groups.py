import pandas as pd
import numpy as np
from pathlib import Path

# Imports
from .data_loader import load_student_data, preprocess_data
from .clustering import (
    compute_feature_vector, 
    compute_distance_matrix,
    enforce_group_size, 
    optimize_gender_balance
)
from .kmedoids import kmedoids_pam_from_matrix


# Configuration & Input
def get_user_target_size() -> int:
    """Handles user input validation."""
    while True:
        try:
            user_input = input("\nEnter target number of students per group (e.g., 5): ")
            size = int(user_input)
            if size <= 0:
                print(" Error: Please enter a number greater than 0.")
                continue
            return size
        except ValueError:
            print("Invalid input. Please enter a whole number.")

def calculate_group_config(n_students: int, target_size: int) -> int:
    """Calculates the optimal number of groups."""
    if target_size > n_students:
        print(f"Target size ({target_size}) > Student count ({n_students}). Creating 1 group.")
        return 1
        
    n_groups = n_students // target_size
    
    # Handle edge case where division results in 0 (should be caught above, but safety first)
    if n_groups < 1: n_groups = 1
    
    print(f"   > Configuration: {n_groups} groups for {n_students} students.")
    return n_groups


# Core Logic (The Pipeline)
def run_grouping_pipeline(df: pd.DataFrame, n_groups: int, target_size: int) -> np.ndarray:
    """
    Orchestrates the Clustering -> Enforcing -> Locking pipeline.
    Returns the final cluster labels.
    """
    # Feature Engineering
    print("   > [Step 1] Computing Features & Distances...")
    feature_matrix = compute_feature_vector(df)
    _, dist_manhattan, _ = compute_distance_matrix(df, feature_matrix)

    # Clustering K-Medoids Manhattan
    print("   > [Step 2] Clustering (K-Medoids Manhattan)...")
    labels, medoid_indices = kmedoids_pam_from_matrix(
        dist_manhattan, 
        n_groups, 
        random_state=42
    )

    # Logistics (Smart Fill)
    print(f"   > [Step 3] Enforcing Group Size (Target: {target_size})...")
    labels = enforce_group_size(
        labels, 
        target_size, 
        feature_matrix=feature_matrix, 
        metric='manhattan' 
    )

    # D. Constraints (Locking System)
    print("   > [Step 4] Optimizing Gender Balance...")
    labels = optimize_gender_balance(
        df, 
        labels, 
        medoid_indices, 
        dist_manhattan
    )
    
    return labels

# ==========================================
# 3. Output & Saving
# ==========================================
def save_results(df: pd.DataFrame, labels: np.ndarray, output_file: str):
    """Formats the dataframe and saves to CSV."""
    df_final = df.copy()
    df_final['Group_ID'] = labels + 1 # 1-based indexing
    
    # Reorder columns
    cols = ['Group_ID'] + [c for c in df_final.columns if c != 'Group_ID']
    df_final = df_final[cols]
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)
    
    print(f"\n SUCCESS! Groups saved to: {output_path}")
    print("\nFinal Group Distribution:")
    print(df_final['Group_ID'].value_counts().sort_index())

# ==========================================
# 4. Main Controller
# ==========================================
def main():
    print("\n" + "="*60)
    print("PART 2: FINAL GROUP GENERATION")
    print("="*60)

    # Configuration
    INPUT_CSV = "backend/data/students_100.csv"
    OUTPUT_CSV = "backend/output/final_groups.csv"

    # 1. Load Data
    print("1. Loading Data...")
    df = load_student_data(INPUT_CSV)
    df = preprocess_data(df)
    
    # 2. Get User Input
    target_size = get_user_target_size()
    n_groups = calculate_group_config(len(df), target_size)

    # 3. Run the "Smart" Logic
    final_labels = run_grouping_pipeline(df, n_groups, target_size)

    # 4. Save
    save_results(df, final_labels, OUTPUT_CSV)

if __name__ == "__main__":
    main()