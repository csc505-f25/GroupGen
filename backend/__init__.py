"""
GroupGen backend — automated student grouping for classroom studies.

Public API for scripts:
  - ``prepare_for_grouping`` — ingest CSV
  - ``run_grouping_pipeline`` — production grouping (returns ``GroupingResult``)
  - ``calculate_n_groups`` — ceil(n / target_size)

See ``docs/ARCHITECTURE.md`` and ``README.md``.
"""

__version__ = "1.0.0"

# Import main functions for easy access
from .data_loader import load_student_data, validate_data, preprocess_data, prepare_for_grouping
from .group_config import calculate_n_groups
from .pipeline import run_grouping_pipeline, GroupingResult

__all__ = [
    'load_student_data',
    'validate_data',
    'preprocess_data',
    'prepare_for_grouping',
    'run_grouping_pipeline',
    'GroupingResult',
    'calculate_n_groups',
]
