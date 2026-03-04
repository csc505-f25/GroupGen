# GroupGen Logic Pipeline

This document explains, step-by-step, how GroupGen converts your raw Google Form CSV into optimized student groups.

## 1. Ingestion (`process_google_form`)
**Input**: Raw CSV file.
**Goal**: Create a clean, structured table of students.
**Logic**:
- **Loads CSV**: Reads the file, skipping any "Ghost Rows" (rows with no Name).
- **Parses Scores**:
    - **Motivation (15 Qs)**: Sums answers (1-4 scale). Total score is mapped to a 1-4 level.
    - **Work Ethic (9 Qs)**: Sums answers (1-4 scale). Total score is mapped to a 1-4 level.
    - **Self-Esteem (8 Qs)**: Averages answers (1-7 scale). Mean is mapped to a 1-4 level.
- **Parses Learning Style**:
    - Scans 28 questions.
    - Detects if answer starts with "A" (Visual), "B" (Auditory), or "C" (Kinesthetic).
    - Assigns the student the majority style (e.g., if mostly As -> "Visual").
- **Output**: A DataFrame with clean columns: `Name`, `Motivation` (1-4), `Work_Ethic` (1-4), `Self_Esteem` (1-4), `Learning_Style` (Visual/Auditory/Kinesthetic).

## 2. Vectorization (`compute_feature_vector`)
**Input**: Clean DataFrame.
**Goal**: Convert student data into numbers the algorithm can understand.
**Logic**:
- **Numerical Features**: Motivation, Work Ethic, Self-Esteem are normalized to a 0-1 scale.
    - *Why?* So that a difference in Motivation (scale 1-4) is treated equally to other features.
- **Categorical Features**: Learning Style, Gender, Diversity are "One-Hot Encoded".
    - *Example*: "Visual" becomes `[1, 0, 0]`, "Auditory" becomes `[0, 1, 0]`.
- **Output**: A numerical matrix where each row is a student and columns represent their traits.

## 3. Distance Calculation (`compute_distance_matrix`)
**Input**: Numerical Matrix.
**Goal**: Measure how "different" every student is from every other student.
**Logic**:
- Uses **Gower Distance** (approximated via Manhattan distance for mixed data).
- Calculates a score (0 to 1) for every pair of students.
    - 0 = Identical students.
    - 1 = Completely opposite students.
- **Output**: A square matrix (NxN) of distances.

## 4. Clustering (K-Medoids / PAM)
**Input**: Distance Matrix.
**Goal**: Find the "centers" (Medoids) of the groups.
**Logic**:
- **Algorithm**: Partitioning Around Medoids (PAM).
- Randomly picks `k` students to be group leaders (Medoids).
- Assigns every other student to the nearest leader based on the Distance Matrix.
- Iteratively swaps leaders to minimize the total distance (finding the most representative centers).
- **Result**: Initial groups based purely on similarity, but sizes might be uneven (e.g., one huge group, one small group).
- **Output**: Initial cluster labels for each student.

## 5. Balancing (`enforce_group_size`)
**Input**: Initial Clusters.
**Goal**: Ensure every group has exactly 4 or 5 students (or user-defined size).
**Logic**:
- Identifies "Donor" groups (too big) and "Receiver" groups (too small).
- Moves students from Donor to Receiver.
    - *Smart Move*: It picks the student who is "least happy" in the Donor group (furthest from center) and moves them to the Receiver group where they fit best.
- Repeats until all groups are within ±1 of the desired size.
- **Output**: Final, balanced groups.
