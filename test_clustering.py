import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from backend.data_loader import load_student_data, preprocess_data
from backend.clustering import (
    compute_feature_vector,
    compute_distance_matrix,
    enforce_group_size,
    check_gender_isolation,
    fix_gender_isolation,
    check_diversity_isolation,
    fix_diversity_isolation,
)
from backend.kmedoids import kmedoids_pam

SKILL_COLS = ["Motivation", "Self_Esteem", "Work_Ethic"]

def strict_gender_alone_violation(group_df: pd.DataFrame) -> bool:
    males = (group_df["Gender"] == "Male").sum()
    females = (group_df["Gender"] == "Female").sum()
    return males == 0 or females == 0

def group_skill_stats(group_df: pd.DataFrame) -> float:
    # mean variance across the 3 skill columns (lower => more homogeneous)
    v = group_df[SKILL_COLS].var()
    return float(v.mean())

def print_groups_one_by_one(stage_name: str, df: pd.DataFrame, labels: np.ndarray):
    tmp = df.copy()
    tmp["Group_ID"] = labels + 1

    print("\n" + "=" * 110)
    print(f"STAGE: {stage_name}")
    print("=" * 110)

    group_ids = sorted(tmp["Group_ID"].unique())
    print(f"Total students: {len(tmp)} | Total groups: {len(group_ids)}")
    print("Group sizes:", tmp["Group_ID"].value_counts().sort_index().to_dict())

    # also show how many groups your *code rule* flags (1-vs-many only)
    code_gender_isolated = check_gender_isolation(tmp, labels)
    code_div_isolated = check_diversity_isolation(tmp, labels)
    print(f"Your code flags gender-isolated groups (1 vs many only): {len(code_gender_isolated)} -> {list(code_gender_isolated.keys())}")
    print(f"Your code flags diversity-isolated groups: {len(code_div_isolated)} -> {list(code_div_isolated.keys())}")

    for gid in group_ids:
        g = tmp[tmp["Group_ID"] == gid].copy()

        males = (g["Gender"] == "Male").sum()
        females = (g["Gender"] == "Female").sum()
        violation = strict_gender_alone_violation(g)
        mean_skill_var = group_skill_stats(g)

        print("\n" + "-" * 110)
        print(f"GROUP {gid}")
        print(f"  Members: {len(g)} | Male={males} | Female={females} | Strict gender-alone violation: {'YES' if violation else 'NO'}")
        print(f"  Mean within-group skill variance (Mot/Self/Work): {mean_skill_var:.4f}")

        # student listing
        # (keeps it readable; you can remove Diversity/Style if too long)
        for _, row in g.sort_values("Name").iterrows():
            print(f"   - {row['Name']} | {row['Gender']} | {row['Diversity']} | {row['Learning_Style']}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, default="backend/data/actual_students.csv")
    ap.add_argument("--target_size", type=int, default=5)
    ap.add_argument("--kmedoids_seed", type=int, default=42)
    args = ap.parse_args()

    df = load_student_data(args.input_csv)
    df = preprocess_data(df)

    # Mirror generate_groups.py behavior
    n_students = len(df)
    n_groups = max(1, n_students // args.target_size)

    # Step 1: features + distances
    print("Computing features & Manhattan/Euclidean distances...")
    feature_matrix = compute_feature_vector(df)
    euc_dist, dist_manhattan, _ = compute_distance_matrix(df, feature_matrix)

    # A) After K-Medoids Manhattan
    print("\nRunning K-Medoids Manhattan...")
    labels_A, _ = kmedoids_pam(dist_manhattan, n_groups, random_state=args.kmedoids_seed)

    # B) After enforce_group_size (Smart Fill)
    print("\nEnforcing group sizes...")
    labels_B = enforce_group_size(
        labels_A,
        args.target_size,
        feature_matrix=feature_matrix,
        metric="manhattan",
    )

    # C) After gender locking (your current rule)
    print("\nApplying gender locking (your code's rule)...")
    isolated_gender = check_gender_isolation(df, labels_B)
    labels_C = labels_B
    if isolated_gender:
        labels_C = fix_gender_isolation(df, labels_B, euc_dist, isolated_gender)

    # D) After diversity locking
    print("\nApplying diversity locking...")
    isolated_div = check_diversity_isolation(df, labels_C)
    labels_D = labels_C
    if isolated_div:
        labels_D = fix_diversity_isolation(df, labels_C, euc_dist, isolated_div)

    # Print groups one-by-one at each stage
    print_groups_one_by_one("A) After K-Medoids Manhattan", df, labels_A)
    print_groups_one_by_one("B) After enforce_group_size (Smart Fill)", df, labels_B)
    print_groups_one_by_one("C) After gender locking swaps", df, labels_C)
    print_groups_one_by_one("D) After diversity locking swaps", df, labels_D)

if __name__ == "__main__":
    main()