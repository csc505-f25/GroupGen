# GroupGen: Automated Student Grouping System

GroupGen is a machine learning pipeline designed to form student groups that are both skill-balanced and demographically inclusive. It uses unsupervised clustering algorithms to group students based on behavioral traits (Motivation, Work Ethic, Self-Esteem) and applies a custom "Locking Mechanism" to ensure fairness and prevent demographic isolation.

## Project Structure

The project is divided into two distinct phases:

1.  **Evaluation & Research:** A comparative analysis ("Tournament") of K-Means vs. K-Medoids using Euclidean, Manhattan, and Gower distances to determine the optimal algorithm.
2.  **Implementation:** The production pipeline that uses the winning algorithm (K-Medoids Manhattan) to generate actual student groups.

```
GroupGen/
├── backend/
│   ├── __init__.py
│   ├── data_loader.py       # Loads and validates CSV data
│   ├── clustering.py        # Core clustering logic & locking mechanisms
│   ├── kmedoids.py          # Custom K-Medoids (PAM) implementation
│   ├── evaluate_clustering.py # Metrics (Silhouette, Entropy, Variance)
│   ├── run_full_evaluation.py # PART 1: The Evaluation Script
│   ├── generate_groups.py     # PART 2: The Group Generator
│   └── data/
│       └── sample_students100.csv
|       └── sample_students30.csv      # Example data file
├── requirements.txt         # Python dependencies
├── IMPLEMENTATION_GUIDE.md  # Detailed implementation instructions
└── README.md               # This file
```

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/GroupGen.git
cd GroupGen
```


## Install dependencies:

```bash
pip install -r requirements.txt
```

(Requires: numpy, pandas, scikit-learn, matplotlib, gower)

## Part 1: Evaluation & Research

Objective: To scientifically compare clustering algorithms and determine which one best balances structural integrity with demographic diversity.

This script runs a tournament between:

- K-Means (Euclidean)

- K-Means (Manhattan)

- K-Medoids (Manhattan)

- K-Medoids (Gower)

It generates comparison tables, skill variance reports, and visual plots.

How to Run:

```bash
python -m backend.run_full_evaluation

```

### What to Expect:

- The script will prompt you for a Group Size.

- It calculates metrics like Silhouette Score, Calinski-Harabasz Index, and Mean Gender Entropy.

- It saves plots (Bar charts, Scatter plots) to backend/output_plots/.

- It prints a final Comparison Table to the console declaring the performance of each metric.

## Part 2: Group Generation(The Tool)

Objective: To generate the final, optimized student groups for a classroom using the best-performing algorithm (K-Medoids Manhattan).

This script handles real-world logistics:

- Smart Fill: Distributes remainder students (e.g., class size 31, target 5) to their mathematically closest group.

- Locking Mechanism: Automatically swaps students to fix gender isolation without breaking the clusters.

How to Run:

```bash
python -m backend.generate_groups
```

### What to Expect
- The script will prompt you for the Target Group Size.
- It runs the full pipeline
- It saves the final group assignments to: backend/output/final_groups.csv

## Data Format 
The system expects a CSV file with the following columns

| Column | Type | Description |
| :--- | :--- | :--- |
| `Name` | String | Student Identifier |
| `Gender` | String | Male / Female |
| `Motivation` | Int (1-4) | 1=Low, 4=High |
| `Self_Esteem` | Int (1-4) | 1=Low, 4=High |
| `Work_Ethic` | Int (1-4) | 1=Low, 4=High |
| `Learning_Style` | String | Visual, Auditory, Kinesthetic |
| `Diversity` | String | Race/Ethnicity category |

## License

[Your License Here]
