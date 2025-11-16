# Implementation Guide

This guide explains what needs to be implemented in each module.

## Overview

The goal is to cluster students based on:
- **Motivation** (1-4 scale)
- **Self-Esteem** (1-4 scale)
- **Work-Ethic** (1-4 scale)
- **Learning Style** (categorical: Visual, Auditory, Kinesthetic, etc.)

While ensuring:
- **Gender Balance**: No student should be the only one of their gender in a group
- **Diversity Balance**: No student should be the only one of their diversity category in a group

## Data Format

Your CSV file should have these columns:
- `Name`: Student name/ID (string)
- `Gender`: 'M'/'F' or 'Male'/'Female' (string)
- `Motivation`: 1-4 (integer)
- `Self_Esteem`: 1-4 (integer)
- `Work_Ethic`: 1-4 (integer)
- `Learning_Style`: e.g., 'Visual', 'Auditory', 'Kinesthetic' (string)
- `Diversity`: e.g., 'White American', 'Black American', 'Hispanic', etc. (string)

**Note on Scale (1-4)**: The 1-4 scale is used instead of 1-10 to make it easier to convert from binary responses (e.g., Yes/No, Agree/Disagree) that you may receive from actual students in classrooms. This simplifies data collection and preprocessing.

## Implementation Steps

### 1. Data Loading (`data_loader.py`)

**`load_student_data()`**:
- Use `pd.read_csv()` to load the file
- Check if required columns exist (raise error if missing)
- Return DataFrame

**`validate_data()`**:
- Check required columns: Name, Gender, Motivation, Self_Esteem, Work_Ethic, Learning_Style, Diversity
- Validate data types
- Check value ranges (Motivation, Self_Esteem, Work_Ethic should be 1-4)
- Check for missing values
- Return `(True, [])` if valid, `(False, [errors])` if invalid

**`preprocess_data()`**:
- Handle missing values (fill or drop)
- Normalize Gender values (e.g., 'M'/'Male' -> 'Male', 'F'/'Female' -> 'Female')
- Normalize Diversity and Learning_Style categories
- Ensure numeric columns are correct types

### 2. Clustering (`clustering.py`)

**`compute_feature_vector()`**:
- Extract Motivation, Self_Esteem, Work_Ethic (numeric)
- Encode Learning_Style (use LabelEncoder or OneHotEncoder from sklearn)
- Optionally normalize features using StandardScaler or MinMaxScaler
- Return N×M matrix (N students, M features)

**`compute_distance_matrix()`**:
- Compute pairwise distances between students
- Options:
  - Euclidean distance: `from scipy.spatial.distance import pdist, squareform`
  - Or use `sklearn.metrics.pairwise_distances()`
- Return N×N distance matrix

**`initial_clustering()`**:
- Use K-Medoids or K-Means clustering
- Options:
  - `from sklearn_extra.cluster import KMedoids` (install scikit-learn-extra)
  - Or `from sklearn.cluster import KMeans` (uses feature vectors instead of distance matrix)
- Return cluster labels (array of length N)

**`check_gender_isolation()`**:
- For each group, count males and females
- If `(females == 1 and males > 1)`: mark as `'female_isolated'`
- If `(males == 1 and females > 1)`: mark as `'male_isolated'`
- Return dictionary: `{group_id: isolation_type}`

**`check_diversity_isolation()`**:
- For each group, count diversity categories
- For each category, if `(count == 1 and group_size > 1)`: mark as isolated
- Return dictionary: `{group_id: '{category}_isolated'}`

**`fix_gender_isolation()`**:
- For each isolated group:
  - If female isolated: find another group with a female
  - Swap that female with a male from the isolated group
  - OR: Move another female to the isolated group (may need to adjust group sizes)
- Try to minimize distance increase when swapping
- Return updated labels

**`fix_diversity_isolation()`**:
- Similar to gender isolation fixing
- For each isolated group:
  - Find another student of the same diversity category from a different group
  - Move them to the isolated group (or swap)
- Return updated labels

**`form_balanced_groups()`**:
- Main function that orchestrates everything
- Steps:
  1. Compute feature vectors
  2. Compute distance matrix
  3. Initial clustering
  4. Check and fix gender isolation (if enabled)
  5. Check and fix diversity isolation (if enabled)
  6. Add Group column to dataframe
  7. Create groups dictionary
  8. Return results

### 3. Main Script (`main.py`)

**`main()`**:
- Load data using `load_student_data()`
- Validate using `validate_data()`
- Preprocess using `preprocess_data()`
- Form groups using `form_balanced_groups()`
- Display results
- Save to CSV file

## Key Considerations

### Gender/Diversity Locking Strategy

**Option 1: Post-processing (Recommended)**
1. Perform initial clustering based on features
2. Check for isolation
3. Fix isolation by swapping/moving students
4. Try to minimize distance increase

**Option 2: Constraint-based Clustering**
- Modify clustering algorithm to consider constraints
- More complex but potentially better results

### Choosing Students to Swap

When fixing isolation, try to:
1. Find students from groups that won't create new isolation
2. Minimize distance increase (choose closest match)
3. Maintain group balance (don't make groups too large/small)

### Handling Edge Cases

- What if there's only one student of a gender/diversity category?
  - Consider: Keep them in a group, or mark as special case
- What if groups become unbalanced after fixing isolation?
  - May need to adjust multiple groups
- What if fixing one isolation creates another?
  - May need iterative fixing

## Testing

Create test cases:
1. Small dataset (10-20 students)
2. Dataset with clear gender/diversity imbalances
3. Edge cases (all one gender, all one diversity category)

## Example Workflow

```python
# 1. Load data
df = load_student_data("students.csv")

# 2. Validate
is_valid, errors = validate_data(df)
if not is_valid:
    print(errors)
    exit()

# 3. Preprocess
df_cleaned = preprocess_data(df)

# 4. Form groups
result_df, groups_dict = form_balanced_groups(
    df_cleaned,
    n_groups=5,
    enforce_gender_balance=True,
    enforce_diversity_balance=True
)

# 5. Check results
print(result_df.groupby('Group')['Gender'].value_counts())
print(result_df.groupby('Group')['Diversity'].value_counts())
```

## Next Steps

1. Implement data loading and validation
2. Implement feature vector computation
3. Implement basic clustering
4. Implement isolation detection
5. Implement isolation fixing
6. Test and refine

