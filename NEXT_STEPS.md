# Implementation Roadmap

## Step 1: Fix Import Issues ✅

First, fix the imports in `main.py`. The imports need to reference the backend package correctly.

## Step 2: Implement Data Loading (Start Here!) 📥

**File: `backend/data_loader.py`**

This is the easiest place to start. Implement these functions in order:

### 2.1 `load_student_data()`
- Use `pd.read_csv(filepath)` to load the CSV
- Check if required columns exist (Name, Gender, Motivation, Self_Esteem, Work_Ethic, Learning_Style, Diversity)
- Raise an error if columns are missing
- Return the DataFrame

**Hint:**
```python
df = pd.read_csv(filepath)
required_cols = ['Name', 'Gender', 'Motivation', 'Self_Esteem', 'Work_Ethic', 'Learning_Style', 'Diversity']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")
return df
```

### 2.2 `validate_data()`
- Check if DataFrame is empty
- Check for required columns
- Check data types (Motivation, Self_Esteem, Work_Ethic should be numeric)
- Check value ranges (1-4 for numeric columns)
- Check for missing values
- Return `(True, [])` if valid, `(False, [list of errors])` if not

**Hint:**
```python
errors = []
if df.empty:
    errors.append("DataFrame is empty")
# Check columns, types, ranges, missing values
return (len(errors) == 0, errors)
```

### 2.3 `preprocess_data()`
- Handle missing values (fill with mode/median or drop)
- Normalize Gender: 'M'/'Male' → 'Male', 'F'/'Female' → 'Female'
- Normalize categorical values (trim whitespace, standardize case)
- Convert numeric columns to int type
- Return cleaned DataFrame

## Step 3: Implement Basic Clustering 🎯

**File: `backend/clustering.py`**

### 3.1 `compute_feature_vector()`
- Extract numeric columns: Motivation, Self_Esteem, Work_Ethic
- Encode Learning_Style using LabelEncoder from sklearn
- Optionally normalize using StandardScaler
- Combine into feature matrix
- Return numpy array

**Hint:**
```python
from sklearn.preprocessing import LabelEncoder, StandardScaler

numeric_features = df[['Motivation', 'Self_Esteem', 'Work_Ethic']].values
le = LabelEncoder()
learning_style_encoded = le.fit_transform(df['Learning_Style']).reshape(-1, 1)
features = np.hstack([numeric_features, learning_style_encoded])
scaler = StandardScaler()
features = scaler.fit_transform(features)
return features
```

### 3.2 `compute_distance_matrix()`
- Use Euclidean distance or Manhattan distance
- Use `sklearn.metrics.pairwise_distances()` or `scipy.spatial.distance.pdist()`
- Return N×N distance matrix

**Hint:**
```python
from sklearn.metrics import pairwise_distances
distance_matrix = pairwise_distances(feature_matrix, metric='euclidean')
return distance_matrix
```

### 3.3 `initial_clustering()`
- Use K-Means or K-Medoids
- For K-Means: `from sklearn.cluster import KMeans`
- For K-Medoids: `from sklearn_extra.cluster import KMedoids` (or implement your own)
- Return cluster labels (array of length N)

**Hint:**
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
labels = kmeans.fit_predict(feature_matrix)  # or distance_matrix for K-Medoids
return labels
```

## Step 4: Implement Isolation Detection 🔍

### 4.1 `check_gender_isolation()`
- For each group (unique label), count males and females
- If females == 1 and males > 1: mark as 'female_isolated'
- If males == 1 and females > 1: mark as 'male_isolated'
- Return dictionary: `{group_id: isolation_type}`

**Hint:**
```python
isolated = {}
for group_id in np.unique(labels):
    group_data = df[labels == group_id]
    male_count = (group_data['Gender'] == 'Male').sum()
    female_count = (group_data['Gender'] == 'Female').sum()
    if female_count == 1 and male_count > 1:
        isolated[group_id] = 'female_isolated'
    elif male_count == 1 and female_count > 1:
        isolated[group_id] = 'male_isolated'
return isolated
```

### 4.2 `check_diversity_isolation()`
- Similar to gender isolation
- For each group, count diversity categories
- If any category has count == 1 and group_size > 1: mark as isolated
- Return dictionary: `{group_id: '{category}_isolated'}`

## Step 5: Implement Isolation Fixing 🔧

### 5.1 `fix_gender_isolation()`
- For each isolated group:
  - Find another student of the same gender from a different group
  - Swap them with a student of the opposite gender from the isolated group
  - Try to minimize distance increase
- Return updated labels

**Strategy:**
1. Find the isolated student
2. Find all students of the same gender in other groups
3. For each candidate, compute the distance cost of swapping
4. Choose the swap with minimum cost
5. Perform the swap

### 5.2 `fix_diversity_isolation()`
- Similar to gender isolation fixing
- Find students of the same diversity category
- Swap or move to prevent isolation

## Step 6: Implement Main Function 🎬

### 6.1 `form_balanced_groups()`
- Call `compute_feature_vector()`
- Call `compute_distance_matrix()`
- Call `initial_clustering()`
- If `enforce_gender_balance`: check and fix gender isolation
- If `enforce_diversity_balance`: check and fix diversity isolation
- Add 'Group' column to dataframe (labels + 1)
- Create groups dictionary
- Return results

### 6.2 `main()` in `main.py`
- Load data
- Validate data
- Preprocess data
- Form groups
- Display and save results

## Step 7: Testing 🧪

1. Test with `sample_students.csv`
2. Verify groups are balanced
3. Verify no gender/diversity isolation
4. Check that groups have reasonable size distribution

## Recommended Order

1. ✅ Fix imports in `main.py`
2. 📥 Implement `load_student_data()` - Test it works
3. 📥 Implement `validate_data()` - Test it works
4. 📥 Implement `preprocess_data()` - Test it works
5. 🎯 Implement `compute_feature_vector()` - Test it works
6. 🎯 Implement `compute_distance_matrix()` - Test it works
7. 🎯 Implement `initial_clustering()` - Test it works
8. 🔍 Implement `check_gender_isolation()` - Test it works
9. 🔍 Implement `check_diversity_isolation()` - Test it works
10. 🔧 Implement `fix_gender_isolation()` - Test it works
11. 🔧 Implement `fix_diversity_isolation()` - Test it works
12. 🎬 Implement `form_balanced_groups()` - Test full pipeline
13. 🎬 Implement `main()` - Test end-to-end

## Quick Test Commands

```python
# Test data loading
from backend.data_loader import load_student_data
df = load_student_data("sample_students.csv")
print(df.head())

# Test validation
from backend.data_loader import validate_data
is_valid, errors = validate_data(df)
print(is_valid, errors)

# Test preprocessing
from backend.data_loader import preprocess_data
df_clean = preprocess_data(df)
print(df_clean.head())
```

## Getting Help

- Check `IMPLEMENTATION_GUIDE.md` for detailed explanations
- Test each function as you implement it
- Use print statements to debug
- Start simple, then add complexity

