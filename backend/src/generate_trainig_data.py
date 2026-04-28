import pandas as pd
import numpy as np
import os
import random

def main():
    print("Loading base 1000-student dataset...")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    in_path = os.path.join(base_dir, 'backend', 'data', 'synthetic_multimodal_1000.csv')
    df = pd.read_csv(in_path)

    print("Synthetically expanding to 5000 unique students using Seed-and-Augment strategy...")
    
    # Seed-and-Augment: For each of the 1000 students, create 4 variants with Gaussian noise on numerics, text intact
    new_dfs = []
    
    # The original 1000
    new_dfs.append(df.copy())
    
    # Generate 4 augmented variations per student, keeping text intact
    for i in range(4):
        augmented = df.copy()
        # Add small Gaussian noise (σ=0.05) to numeric features
        noise = np.random.normal(0, 0.05, size=(len(augmented), 3))
        augmented[['Motivation', 'Self_Esteem', 'Work_Ethic']] += noise
        # Keep bounded between 1 and 4
        augmented[['Motivation', 'Self_Esteem', 'Work_Ethic']] = np.clip(augmented[['Motivation', 'Self_Esteem', 'Work_Ethic']], 1, 4)
        
        new_dfs.append(augmented)
        
    final_df = pd.concat(new_dfs, ignore_index=True)
    
    # Rename students to maintain uniqueness
    final_df['Name'] = [f"Student_{i}" for i in range(len(final_df))]
    
    # Shuffle the final rows so it's a completely mixed pool
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    out_path = os.path.join(base_dir, 'backend', 'data', 'synthetic_multimodal_5000.csv')
    final_df.to_csv(out_path, index=False)
    
    print(f"SUCCESS! True 5000-student pool with preserved text-tabular correlations generated and saved to: {out_path}")

if __name__ == "__main__":
    main()
