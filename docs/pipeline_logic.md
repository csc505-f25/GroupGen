# GroupGen Logic Pipeline

This document explains, step-by-step, how GroupGen converts your raw Google Form CSV into optimized student groups.

## 1. Ingestion (`group_gen_intake.py`)
**Input**: Raw CSV file.
**Goal**: Create a clean, structured table of students.
**Logic**:
- **Loads CSV & Detects Boundaries**: Reads the file and uses the repeated "Magic Wand" questions as unbreakable anchors. It calculates the column indexes of these wands to dynamically slice the exact boundaries for the Self-Esteem, Motivation, and Work Ethic blocks, making the script completely immune to added or deleted survey questions.
- **Parses Scores**:
    - Skips "Ghost Rows" (rows with an empty Name field).
    - Uses a robust regex helper to parse text answers intelligently:
        - Prioritizes extracting digits directly from prefixed strings (e.g. "5 - Strongly Agree" -> 5).
        - If no digits exist, maps common Likert agreement scales ("Strongly Agree" -> 5, "Disagree" -> 2) AND Likert frequency scales ("Always" -> 4, "Usually" -> 3) to standard numeric floats.
    - **Motivation (15 Qs)** & **Work Ethic (9 Qs)**: Mean scores are computed and clamped into a `1-4` level scale.
    - **Self-Esteem (8 Qs)**: Mean scores are computed (from a 1-7 scale) and converted into a `1-4` level scale.
- **Parses Learning Style**:
    - Looks at the answers to learning style questions (Options starting with A, B, or C).
    - Assigns the student the majority style ("Visual", "Auditory", or "Kinesthetic").
- **Output**: A clean DataFrame with `Name`, `Gender`, `Diversity`, `Learning_Style`, `Motivation`, `Self_Esteem`, `Work_Ethic`.

## 2. Vectorization (`compute_feature_vector`)
**Input**: Clean DataFrame.
**Goal**: Convert student data into numbers the algorithm can understand.
**Logic**:
- **Numeric Features**: Motivation, Work Ethic, and Self-Esteem are extracted.
- **Categorical Features**: Learning Style is transformed using One-Hot Encoding (e.g. "Visual" -> `[1, 0, 0]`).
- **Standardization**: All features are scaled using `StandardScaler` (Mean = 0, Variance = 1).
    - *Why?* This ensures that traits operating on small scales don't outweigh traits on larger scales, preventing algorithms from overly indexing on a single variable.
- **Output**: A standardized numerical matrix.

## 3. Distance Calculation (`compute_distance_matrix`)
**Input**: Standardized Numerical Matrix.
**Goal**: Measure how "different" every student is from every other student.
**Logic**:
- Computes multiple distance metrics, but the system **actively uses Manhattan Distance** (`metric='manhattan'`).
- Calculates distance between all student pairs to feed into the clustering algorithm.
- **Output**: A square NxN distance matrix.

## 4. Initial Clustering (`kmedoids_pam`)
**Input**: Manhattan Distance Matrix.
**Goal**: Form initial group clusters strictly based on similarity features.
**Logic**:
- The total Number of Groups (`K`) is dynamically calculated by dividing total students by `group_size`.
- **Algorithm**: Partitioning Around Medoids (PAM).
    - Selects `K` students to act as initial "leaders" (Medoids).
    - Assigns every other student to the closest Medoid based on the Manhattan distance.
    - Swaps leaders iteratively until the grouping is mathematically optimized for homogeneity (similar trait clusters).
- **Result**: Initial clusters, but their sizes might be wildly unbalanced at this stage.
- **Output**: An array of cluster labels.

## 5. Balancing (`enforce_group_size`)
**Input**: Unbalanced Clusters & Feature Matrix.
**Goal**: Ensure every group matches the user-defined `group_size` (e.g., 4 or 5 students).
**Logic**:
- Re-calculates exact target capacities for each group, accounting for remainders.
- Identifies **Donor** groups (meaning they have too many students) and **Receiver** groups (too few students).
- Examines *every* student in a Donor group against *every* Receiver group's center point.
- Finds the globally optimum single swap (the Donor student closest to a Receiver center) and moves them.
- Repeats until all clusters hit their exact target capacity constraints.
- **Output**: Perfectly size-balanced clusters.

## 6. Gender Constraint Check (`check_gender_isolation` / `fix_gender_isolation`)
**Input**: Balanced Clusters.
**Goal**: Prevent gender isolation (e.g., 1 woman in a table of 4 men).
**Logic**:
- **Detection**: Loops through all groups > 3 members to see if there is exactly 1 male or exactly 1 female.
- **Resolution**: If an isolation event is found:
    - It searches for a "donor" group that either has `>2` students of that gender, OR exactly `1` of that gender (combining them).
    - It evaluates distances to safely swap the isolated student out (or swap an ally student in) without creating a new isolation event in the process, choosing the minimum feature distance cost.
- **Output**: Clusters modified to distribute genders fairly and eliminate 1-person isolations.

## 7. Diversity Constraint Check (`check_diversity_isolation` / `fix_diversity_isolation`)
**Input**: Gender-Balanced Clusters.
**Goal**: Prevent marginalized group isolation by pairing.
**Logic**:
- **Detection**: Loops through groups > 2 members checking the `Diversity` identity column. If any identity label appears exactly once in the group, it is flagged as an isolated diversity scenario.
- **Resolution**: Search other groups for a donor group that contains `>1` of that specific diversity category.
    - Swaps another student of the matching identity into the isolated group, while swapping out a student of a different label.
    - Proximity/Distance is optimized to preserve general similarity as closely as possible during the swap.
- **Output**: Final group assignments that factor in similarity, size, gender distribution, and diversity.

## 8. Phase 1: Evaluation Layer (`evaluate_clustering.py` & `run_full_evaluation.py`)
**Input**: Standardized Feature Matrix & Label Results.
**Goal**: Mathematically validate the quality of clustering prior to constraint-locking.
**Logic**:
- **Algorithm Tournament**: Executes K-Means (Euclidean), K-Means (Manhattan), K-Medoids (Manhattan), and K-Medoids (Gower) side-by-side to compare performance quantitatively on the dataset.
- **Scoring Metrics Evaluated**:
    - **Silhouette Score**: Measures cluster cohesion and separation (-1 to 1). High score means students fit their assigned group significantly better than neighboring groups.
    - **Davies-Bouldin Index**: Measures average similarity ratio (lower is better).
    - **Calinski-Harabasz Index**: Measures the ratio of between-cluster to within-cluster dispersion (higher is better).
    - **Mean Gender Entropy**: Measures how naturally balanced the cluster was without forced overrides (lower entropy means more balanced).
- **Output**: Comparison CSVs and visual Plotly/Matplotlib PNGs saved to the `output_plots` directory to arm administrators with empirical evidence of grouping effectiveness.
