"""
Data Loading Module

This module handles loading student data from CSV files.
You need to implement functions to:
1. Load CSV data
2. Validate required columns exist
3. Clean/preprocess the data
4. Handle missing values
"""

import pandas as pd
from typing import Optional, List, Dict, Tuple


def load_student_data(filepath: str) -> pd.DataFrame:
    """
    Load student data from a CSV file.
    
    Expected columns:
    - Name (string): Student name/ID
    - Gender (string): 'M' or 'F' or 'Male'/'Female'
    - Motivation (int): 1-4 scale
    - Self_Esteem (int): 1-4 scale
    - Work_Ethic (int): 1-4 scale
    - Learning_Style (string): e.g., 'Visual', 'Auditory', 'Kinesthetic'
    - Diversity (string): e.g., 'White American', 'Black American', 'Hispanic', etc.
    """
    required_columns = [
            'Name',
            'Gender',
            'Motivation',
            'Self_Esteem',
            'Work_Ethic',
            'Learning_Style',
            'Diversity'
    ]
    
     # Load the CSV
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error loading CSV: {e}")

    # Validate required columns
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df



def validate_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that the DataFrame has all required columns and valid data.
    
    Returns:
        (is_valid, list_of_errors)
    """
    
    errors = []
    required_cols = [
        'Name',
        'Gender',
        'Motivation',
        'Self_Esteem',
        'Work_Ethic',
        'Learning_Style',
        'Diversity'
    ]
    
    # Check required columns
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")

    # Check numeric ranges
    for col in ["Motivation", "Self_Esteem", "Work_Ethic"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"{col} must be numeric.")
        elif not df[col].between(1, 4).all():
            errors.append(f"{col} contains values outside the 1–4 range.")

    # Check for missing values
    if df.isnull().any().any():
        errors.append("Data contains missing values.")

    return (len(errors) == 0, errors)



def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the student data.
    """
    
    # Handle missing values (drop rows with missing data)
    df = df.dropna()

    # Normalize Gender
    df["Gender"] = df["Gender"].str.strip().str.capitalize()
    df["Gender"] = df["Gender"].replace({
        "M": "Male",
        "F": "Female"
    })

    # Normalize Learning Style
    df["Learning_Style"] = df["Learning_Style"].str.strip().str.capitalize()

    # Standardize Diversity categories
    df["Diversity"] = df["Diversity"].str.strip().str.title()

    # Ensure numeric columns are numeric
    numeric_cols = ["Motivation", "Self_Esteem", "Work_Ethic"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    return df

