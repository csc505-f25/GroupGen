# GroupGen

A Python project that automatically forms balanced and diverse student groups using clustering, with special locking mechanisms to prevent gender and diversity isolation.

## Features

- **Feature-based Clustering**: Groups students based on Motivation, Self-Esteem, Work-Ethic, and Learning Style
- **Gender Balance Locking**: Ensures no student is the only one of their gender in a group
- **Diversity Balance Locking**: Ensures no student is the only one of their diversity category in a group
- **Modular Design**: Clean skeleton code structure for easy implementation

## Project Structure

```
GroupGen/
├── backend/
│   ├── __init__.py
│   ├── data_loader.py      # Data loading and validation (TODO: implement)
│   ├── clustering.py        # Clustering with locking mechanisms (TODO: implement)
│   └── main.py              # Main orchestration script (TODO: implement)
├── sample_students.csv      # Example data file
├── requirements.txt         # Python dependencies
├── IMPLEMENTATION_GUIDE.md  # Detailed implementation instructions
└── README.md               # This file
```

## Data Format

Your CSV file should have these columns:
- `Name`: Student name/ID
- `Gender`: 'M'/'F' or 'Male'/'Female'
- `Motivation`: 1-4 scale
- `Self_Esteem`: 1-4 scale
- `Work_Ethic`: 1-4 scale
- `Learning_Style`: e.g., 'Visual', 'Auditory', 'Kinesthetic'
- `Diversity`: e.g., 'White American', 'Black American', 'Hispanic', etc.

See `sample_students.csv` for an example.

**Note on Scale**: The 1-4 scale for Motivation, Self-Esteem, and Work-Ethic is used to make it easier to convert from binary responses (Yes/No, Agree/Disagree) that you may receive from actual students in classrooms.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Implementation Status

This is a **skeleton codebase** with TODO comments. You need to implement:

1. **Data Loading** (`backend/data_loader.py`):
   - Load CSV files
   - Validate data
   - Preprocess data

2. **Clustering** (`backend/clustering.py`):
   - Compute feature vectors
   - Compute distance matrices
   - Perform initial clustering
   - Detect gender/diversity isolation
   - Fix isolation issues

3. **Main Script** (`backend/main.py`):
   - Orchestrate the workflow
   - Display results

See `IMPLEMENTATION_GUIDE.md` for detailed instructions.

## Usage (After Implementation)

```python
from backend import load_student_data, form_balanced_groups

# Load data
df = load_student_data("students.csv")

# Form groups
result_df, groups_dict = form_balanced_groups(
    df,
    n_groups=5,
    enforce_gender_balance=True,
    enforce_diversity_balance=True
)

# View results
print(result_df[['Name', 'Group', 'Gender', 'Diversity']])
```

## Key Concepts

### Gender/Diversity Locking

The algorithm prevents isolation by:
1. Performing initial clustering based on features
2. Detecting groups where one student is isolated (e.g., one female among many males)
3. Fixing isolation by swapping or moving students to ensure balance

### Clustering Features

Students are clustered based on:
- **Motivation** (1-4)
- **Self-Esteem** (1-4)
- **Work-Ethic** (1-4)
- **Learning Style** (categorical, encoded as numeric)

## Next Steps

1. Read `IMPLEMENTATION_GUIDE.md` for detailed instructions
2. Implement the functions marked with `TODO` comments
3. Test with `sample_students.csv`
4. Refine the locking mechanisms as needed

## License

[Your License Here]
