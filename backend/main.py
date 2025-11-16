"""
Main Orchestration Script

This script ties everything together to form student groups.
"""

import pandas as pd
from pathlib import Path
from data_loader import load_student_data, validate_data, preprocess_data
from clustering import form_balanced_groups


def main():
    """
    Main function to run the group formation pipeline.
    
    TODO: Implement the main workflow:
    1. Load student data from CSV
    2. Validate the data
    3. Preprocess the data
    4. Form balanced groups with locking mechanisms
    5. Display results and save to file
    """
    # Step 1: Load data
    #input_file = "data/sample_students.csv"
    input_file = Path(__file__).resolve().parent / "data" / "sample_students.csv"
   
    df = load_student_data(str(input_file)) #First thing to do
    
    #TODO: Step 2: Validate data
    is_valid, errors = validate_data(df)
    if not is_valid:
        print("Data validation failed:")
        for error in errors:
            print(f"  - {error}")
        return
    
    # TODO: Step 3: Preprocess data
    df_cleaned = preprocess_data(df)

    #Final Processed data output
    # print("\n=== FINAL PREPROCESSED DATA ===")
    # print(df_cleaned.head())    # show first 5 rows
    # print("\n=== DATA TYPES ===")
    # print(df_cleaned.dtypes)
    # print("\n=== UNIQUE VALUES CHECK ===")
    # print("Gender:", df_cleaned["Gender"].unique())
    # print("Learning Style:", df_cleaned["Learning_Style"].unique())
    # print("Diversity:", df_cleaned["Diversity"].unique())

    
    # TODO: Step 4: Form groups
    n_groups = 4  # Change this to desired number of groups
    result_df, groups_dict = form_balanced_groups(
        df_cleaned,
        n_groups=n_groups,
        random_state=42,
        enforce_gender_balance=True,
        enforce_diversity_balance=True
    )
    
    # TODO: Step 5: Display and save results
    # print("Group Assignments:")
    # print(result_df[['Name', 'Group', 'Gender', 'Diversity', 'Motivation', 'Self_Esteem', 'Work_Ethic']])
    # 
    # print("\nGroup Composition:")
    # for group_id, students in groups_dict.items():
    #     group_data = result_df[result_df['Group'] == group_id]
    #     print(f"\nGroup {group_id} ({len(students)} students):")
    #     print(f"  Students: {', '.join(students)}")
    #     print(f"  Gender: {group_data['Gender'].value_counts().to_dict()}")
    #     print(f"  Diversity: {group_data['Diversity'].value_counts().to_dict()}")
    #     print(f"  Avg Motivation: {group_data['Motivation'].mean():.2f}")
    #     print(f"  Avg Self-Esteem: {group_data['Self_Esteem'].mean():.2f}")
    #     print(f"  Avg Work-Ethic: {group_data['Work_Ethic'].mean():.2f}")
    # 
    # # Save results
    # output_file = "group_assignments.csv"
    # result_df.to_csv(output_file, index=False)
    # print(f"\nResults saved to {output_file}")
    
    print("TODO: Implement the main() function following the comments above")

if __name__ == "__main__":
    main()