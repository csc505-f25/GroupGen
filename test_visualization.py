"""
Test script to verify visualization works with K-Means clustering.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.clustering import form_balanced_groups, compute_feature_vector, initial_clustering

# Create sample student data for testing
print("Creating sample student data...")
np.random.seed(42)
sample_data = {
    'Name': [f'Student_{i:03d}' for i in range(30)],
    'Gender': ['Male' if i % 2 == 0 else 'Female' for i in range(30)],
    'Motivation': np.random.randint(1, 5, 30),
    'Self_Esteem': np.random.randint(1, 5, 30),
    'Work_Ethic': np.random.randint(1, 5, 30),
    'Learning_Style': np.random.choice(['Visual', 'Auditory', 'Kinesthetic'], 30),
    'Diversity': np.random.choice(['White American', 'Black American', 'Hispanic', 'Asian American'], 30)
}

df = pd.DataFrame(sample_data)
print(f"Created {len(df)} student records")
print("\nSample data:")
print(df.head())

# Test visualization with form_balanced_groups
print("\n" + "="*60)
print("Testing visualization with form_balanced_groups...")
print("="*60)
print("This will show multiple visualization plots.")
print("Close each plot window to continue to the next one.\n")

try:
    result_df, groups_dict = form_balanced_groups(
        df,
        n_groups=4,
        random_state=42,
        enforce_gender_balance=False,  # Disable for now since not implemented
        enforce_diversity_balance=False,  # Disable for now since not implemented
        visualize=True  # Enable visualization
    )
    print("\n[OK] Visualization completed!")
    print(f"     Created {len(groups_dict)} groups")
    print(f"     Group sizes: {result_df['Group'].value_counts().to_dict()}")
except Exception as e:
    print(f"[FAIL] Visualization failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Test complete!")
print("="*60)
print("\nNote: If you see multiple plot windows, close each one to continue.")
print("The visualization shows:")
print("  1. PCA visualization of clusters")
print("  2. Feature pairs (Motivation vs Self-Esteem, etc.)")
print("  3. Cluster size distribution")
print("  4. Clusters with centroids marked")

