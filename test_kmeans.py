"""
Test script to verify K-Means clustering implementation works with student data.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.clustering import compute_feature_vector, kmeans_custom, initial_clustering

# Create sample student data for testing
print("Creating sample student data...")
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

# Test feature vector computation
print("\n" + "="*60)
print("Testing feature vector computation...")
print("="*60)
try:
    feature_matrix = compute_feature_vector(df)
    print(f"[OK] Feature matrix shape: {feature_matrix.shape}")
    print(f"     Expected: ({len(df)}, 4) - (students, features)")
    print(f"     Features: Motivation, Self_Esteem, Work_Ethic, Learning_Style")
    print(f"     Sample feature vector (first student): {feature_matrix[0]}")
except Exception as e:
    print(f"[FAIL] Feature vector computation failed: {e}")
    import traceback
    traceback.print_exc()

# Test K-Means clustering
print("\n" + "="*60)
print("Testing K-Means clustering...")
print("="*60)
try:
    n_clusters = 5
    labels = kmeans_custom(feature_matrix, n_clusters, random_state=42)
    print(f"[OK] Clustering completed!")
    print(f"     Number of clusters: {len(np.unique(labels))}")
    print(f"     Cluster sizes: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print(f"     Labels: {labels[:10]}... (showing first 10)")
except Exception as e:
    print(f"[FAIL] K-Means clustering failed: {e}")
    import traceback
    traceback.print_exc()

# Test initial_clustering function
print("\n" + "="*60)
print("Testing initial_clustering function...")
print("="*60)
try:
    labels = initial_clustering(feature_matrix, n_clusters=5, random_state=42)
    print(f"[OK] initial_clustering() works!")
    print(f"     Number of clusters: {len(np.unique(labels))}")
    print(f"     Cluster distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
except Exception as e:
    print(f"[FAIL] initial_clustering() failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Test complete!")
print("="*60)

