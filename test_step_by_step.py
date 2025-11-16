"""
Step-by-step testing script

Use this to test each function as you implement it.
Uncomment sections as you complete each function.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# STEP 1: Test Data Loading
# ============================================================================
print("=" * 60)
print("STEP 1: Testing Data Loading")
print("=" * 60)

try:
    from backend.data_loader import load_student_data
    df = load_student_data("sample_students.csv")
    print("✅ load_student_data() works!")
    print(f"   Loaded {len(df)} students")
    print(f"   Columns: {list(df.columns)}")
    print(f"   First few rows:")
    print(df.head())
except Exception as e:
    print(f"❌ load_student_data() failed: {e}")
    print("   TODO: Implement load_student_data() in backend/data_loader.py")

print("\n")

# ============================================================================
# STEP 2: Test Data Validation
# ============================================================================
print("=" * 60)
print("STEP 2: Testing Data Validation")
print("=" * 60)

try:
    from backend.data_loader import validate_data
    is_valid, errors = validate_data(df)
    if is_valid:
        print("✅ validate_data() works! Data is valid.")
    else:
        print(f"⚠️  validate_data() works but found errors: {errors}")
except Exception as e:
    print(f"❌ validate_data() failed: {e}")
    print("   TODO: Implement validate_data() in backend/data_loader.py")

print("\n")

# ============================================================================
# STEP 3: Test Data Preprocessing
# ============================================================================
print("=" * 60)
print("STEP 3: Testing Data Preprocessing")
print("=" * 60)

try:
    from backend.data_loader import preprocess_data
    df_clean = preprocess_data(df)
    print("✅ preprocess_data() works!")
    print(f"   Cleaned data shape: {df_clean.shape}")
    print(f"   Gender values: {df_clean['Gender'].unique()}")
    print(f"   First few rows:")
    print(df_clean.head())
except Exception as e:
    print(f"❌ preprocess_data() failed: {e}")
    print("   TODO: Implement preprocess_data() in backend/data_loader.py")

print("\n")

# ============================================================================
# STEP 4: Test Feature Vector Computation
# ============================================================================
print("=" * 60)
print("STEP 4: Testing Feature Vector Computation")
print("=" * 60)

try:
    from backend.clustering import compute_feature_vector
    feature_matrix = compute_feature_vector(df_clean)
    print("✅ compute_feature_vector() works!")
    print(f"   Feature matrix shape: {feature_matrix.shape}")
    print(f"   First few feature vectors:")
    print(feature_matrix[:3])
except Exception as e:
    print(f"❌ compute_feature_vector() failed: {e}")
    print("   TODO: Implement compute_feature_vector() in backend/clustering.py")

print("\n")

# ============================================================================
# STEP 5: Test Distance Matrix Computation
# ============================================================================
print("=" * 60)
print("STEP 5: Testing Distance Matrix Computation")
print("=" * 60)

try:
    from backend.clustering import compute_distance_matrix
    distance_matrix = compute_distance_matrix(feature_matrix)
    print("✅ compute_distance_matrix() works!")
    print(f"   Distance matrix shape: {distance_matrix.shape}")
    print(f"   Average distance: {distance_matrix.mean():.2f}")
except Exception as e:
    print(f"❌ compute_distance_matrix() failed: {e}")
    print("   TODO: Implement compute_distance_matrix() in backend/clustering.py")

print("\n")

# ============================================================================
# STEP 6: Test Initial Clustering
# ============================================================================
print("=" * 60)
print("STEP 6: Testing Initial Clustering")
print("=" * 60)

try:
    from backend.clustering import initial_clustering
    labels = initial_clustering(distance_matrix, n_clusters=5, random_state=42)
    print("✅ initial_clustering() works!")
    print(f"   Number of clusters: {len(set(labels))}")
    print(f"   Cluster sizes: {dict(zip(*np.unique(labels, return_counts=True)))}")
except Exception as e:
    print(f"❌ initial_clustering() failed: {e}")
    print("   TODO: Implement initial_clustering() in backend/clustering.py")

print("\n")

# ============================================================================
# STEP 7: Test Isolation Detection
# ============================================================================
print("=" * 60)
print("STEP 7: Testing Isolation Detection")
print("=" * 60)

try:
    from backend.clustering import check_gender_isolation, check_diversity_isolation
    gender_isolated = check_gender_isolation(df_clean, labels)
    diversity_isolated = check_diversity_isolation(df_clean, labels)
    print("✅ Isolation detection works!")
    print(f"   Gender isolation: {gender_isolated}")
    print(f"   Diversity isolation: {diversity_isolated}")
except Exception as e:
    print(f"❌ Isolation detection failed: {e}")
    print("   TODO: Implement check_gender_isolation() and check_diversity_isolation()")

print("\n")

# ============================================================================
# STEP 8: Test Full Pipeline
# ============================================================================
print("=" * 60)
print("STEP 8: Testing Full Pipeline")
print("=" * 60)

try:
    from backend.clustering import form_balanced_groups
    result_df, groups_dict = form_balanced_groups(
        df_clean,
        n_groups=5,
        random_state=42,
        enforce_gender_balance=True,
        enforce_diversity_balance=True
    )
    print("✅ form_balanced_groups() works!")
    print(f"   Created {len(groups_dict)} groups")
    print(f"   Group sizes: {result_df['Group'].value_counts().to_dict()}")
except Exception as e:
    print(f"❌ form_balanced_groups() failed: {e}")
    print("   TODO: Implement form_balanced_groups() in backend/clustering.py")

print("\n")
print("=" * 60)
print("Testing Complete!")
print("=" * 60)

