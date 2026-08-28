# GroupGen Logic Pipeline

> **Branch `feature-only-clustering`:** Production grouping stops after size balancing. Gender and Diversity are **not** one-hot encoded for clustering and **not** used for post-cluster swaps.

This document explains how GroupGen converts a Google Form CSV into student groups.

## 1. Ingestion (`process_google_form`)
**Input**: Raw CSV file.
**Goal**: Create a clean, structured table of students.
**Logic**:
- **Loads CSV**: Reads the file, skipping blank rows (no Name).
- **Parses Scores**:
    - **Motivation**: Section mean (1–4 Likert) → discrete 1–4 bucket.
    - **Work Ethic**: Section mean (1–4 Likert) → discrete 1–4 bucket.
    - **Self-Esteem**: Section mean (1–7 Likert) → discrete 1–4 bucket.
- **Parses Learning Style**:
    - Scores 30 VAK items via `vak_answer_catalog.py` (A/B/C majority → Visual / Auditory / Kinesthetic).
- **Demographics**: Gender and Diversity are extracted for display but **excluded from clustering** on this branch.
- **Output**: `Name`, `Gender`, `Motivation`, `Self_Esteem`, `Work_Ethic`, `Learning_Style`, `Diversity`.

## 2. Vectorization (`compute_feature_vector`)
**Input**: Clean DataFrame.
**Goal**: Convert behavioral data into a numeric matrix for clustering.
**Logic**:
- **Numerical Features**: Motivation, Self-Esteem, Work Ethic → `StandardScaler` (z-scores).
- **Categorical Feature**: Learning_Style → one-hot (Visual / Auditory / Kinesthetic).
- **Excluded**: Gender, Diversity (never enter the feature matrix in production).
- **Output**: N×M matrix (3 scaled numerics + 3 one-hot columns).

## 3. Distance Calculation (`compute_psychometric_distance_matrix`)
**Input**: Feature matrix.
**Goal**: Pairwise psychometric similarity.
**Logic**:
- **Manhattan (L1)** distance on the feature matrix only.
- **Output**: N×N distance matrix used for clustering and size balancing.

## 4. Clustering (K-Medoids / PAM)
**Input**: Distance matrix.
**Goal**: Assign students to homogeneous skill groups.
**Logic**:
- **Algorithm**: Partitioning Around Medoids (PAM), `random_state=42`.
- Assign each student to the nearest medoid; iteratively swap medoids to reduce total cost.
- **Output**: Initial cluster labels (sizes may be uneven).

## 5. Balancing (`enforce_group_size`)
**Input**: Initial clusters + distance matrix.
**Goal**: Match the `ceil(n / target_size)` size distribution (e.g. groups of 4 and 5).
**Logic**:
- Move students from oversized groups to undersized groups using minimum Manhattan edge cost.
- Repeats until all groups match the expected size pattern or raises an error.
- **Output**: Final, balanced groups.

## 6. Validation (`assert_assignment_invariants`)
**Input**: Final labels.
**Goal**: Guarantee structural correctness before API/CLI returns success.
**Logic**:
- Correct number of groups, every student assigned, no group exceeds `target_size`.

## Legacy (not production on this branch)

- `fairness_distribution.rebalance_demographic_column` — demographic pairing swaps (research/tests only).
- `compute_distance_matrix` + Gower — evaluation scripts only.
