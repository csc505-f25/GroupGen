import pandas as pd
import numpy as np
from pathlib import Path

# Imports
from .data_loader import load_student_data, preprocess_data
from .clustering import (
    compute_feature_vector, compute_distance_matrix, enforce_group_size, check_gender_isolation, fix_gender_isolation, kmeans_custom)
from .kmedoids import kmedoids_pam


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
    labels, medoid_indices = kmedoids_pam(
        dist_manhattan, 
        n_groups, 
        random_state=42
    )
    # labels, centroids = kmeans_custom(
    #     x= feature_matrix,                # REPLACE 'X' with your actual data array
    #     K=n_groups, 
    #     random_state=42,
    #     metric='euclidean',
    #     return_centroids=True
    # )

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
    isolated_g= check_gender_isolation(df, labels)
    labels = fix_gender_isolation(
        df, 
        labels, 
        dist_manhattan,
        isolated_g
    )
    
    return labels

# ==========================================
# 3. Output & Saving
# ==========================================
def save_results(df: pd.DataFrame, labels: np.ndarray, output_file: str):
    """
    1. Saves the raw data to CSV.
    2. IMMEDIATELY RELOADS that CSV to generate the text report.
       (This guarantees the report matches the file exactly).
    """
    # --- STEP 1: PREPARE & SAVE CSV ---
    df_final = df.copy()
    # Force labels to simple integers
    df_final['Group_ID'] = np.array(labels).astype(int) + 1 
    
    # Reorder columns
    cols = ['Group_ID'] + [c for c in df_final.columns if c != 'Group_ID']
    df_final = df_final[cols]
    
    # Sort
    df_final = df_final.sort_values(by=['Group_ID', 'Name'])
    
    # Save 
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)
    
    print(f"\n✅ SUCCESS! Groups saved to: {output_path}")

    # --- STEP 2: THE RELOAD (The Fix) ---
    # We read the file we just saved. This gives us a 100% clean dataframe.
    # No index issues, no hidden data issues.
    print("   > Generating text report from saved CSV...")
    df_clean = pd.read_csv(output_path)

    # --- STEP 3: GENERATE TEXT REPORT ---
    report_path = output_path.with_name(output_path.stem + "_report.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        
        def write_line(text=""):
            print(text)
            f.write(text + "\n")

        write_line("\n" + "="*65)
        write_line("FINAL GROUP ASSIGNMENTS REPORT")
        write_line("="*65)

        # Get the unique Group IDs from the CLEAN dataframe
        unique_groups = sorted(df_clean['Group_ID'].unique())

        for g_id in unique_groups:
            # Filter the CLEAN dataframe
            group_data = df_clean[df_clean['Group_ID'] == g_id]
            
            # Header
            write_line(f"\n GROUP {g_id} ({len(group_data)} students)")
            write_line("-" * 65)
            
            # Column Headers
            header = f"{'Name':<20} | {'Gender':<8} | {'Ethnicity':<20} | {'Style':<10}"
            write_line(header)
            write_line("-" * 65)

            # Rows
            for _, row in group_data.iterrows():
                # Since we reloaded from CSV, these columns are guaranteed to exist as strings/ints
                name = str(row['Name'])
                gender = str(row['Gender'])
                diversity = str(row['Diversity'])
                style = str(row['Learning_Style'])

                # Truncate long strings
                if len(diversity) > 19: diversity = diversity[:17] + ".."

                write_line(f"{name:<20} | {gender:<8} | {diversity:<20} | {style:<10}")
            
            write_line("-" * 65)
            
    print(f"✅ Readable Report saved to: {report_path}")

# ==========================================
# 4. Main Controller
# ==========================================
def main():
    print("\n" + "="*60)
    print("PART 2: FINAL GROUP GENERATION")
    print("="*60)

    # Configuration
    INPUT_CSV = "backend/data/sample_students.csv"
    # INPUT_CSV = "backend/data/actual_students.csv"
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