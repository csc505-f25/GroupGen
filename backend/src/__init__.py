"""
GroupGen Backend Package

Automatic student group formation with gender and diversity balance.
"""

__version__ = "1.0.0"

# Import main functions for easy access
from .data_loader import load_student_data, validate_data, preprocess_data
from .clustering import form_balanced_groups

__all__ = [
    'load_student_data',
    'validate_data',
    'preprocess_data',
    'form_balanced_groups',
]
