"""
Convert NumPy/pandas values to JSON-serializable Python primitives.

Used by ``api.py`` so FastAPI responses never leak ``np.int64``, ``np.nan``,
or ``inf`` to the browser. Recurses dicts, lists, and arrays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _is_non_finite_number(obj) -> bool:
    if isinstance(obj, (np.floating, float)):
        return bool(np.isnan(obj) or np.isinf(obj))
    return False


def to_json_safe(obj):
    """Recursively convert objects for FastAPI/JSON responses."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        # JSON object keys must be strings (value_counts can yield numpy keys).
        return {str(to_json_safe(k)): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if _is_non_finite_number(obj):
            return None  # JSON has no NaN/Infinity
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and _is_non_finite_number(obj):
        return None
    if pd.isna(obj):
        return None
    return obj
