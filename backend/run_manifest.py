"""
Per-run metadata for CLI study audit trail.

Each ``generate_groups`` run writes ``run_manifest.json`` next to the CSV so
researchers can reproduce settings (seed, group size, warnings) without parsing
the roster file.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


def write_run_manifest(
    run_dir: Path,
    *,
    input_path: str,
    n_students: int,
    target_group_size: int,
    n_groups: int,
    random_state: int = 42,
    warnings: Optional[List[str]] = None,
    output_csv: str,
) -> Path:
    """Write run_manifest.json beside CLI outputs."""
    manifest_path = run_dir / "run_manifest.json"
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": input_path,
        "output_csv": output_csv,
        "n_students": n_students,
        "target_group_size": target_group_size,
        "n_groups": n_groups,
        "random_state": random_state,
        "warnings": warnings or [],  # leftover fairness issues, if any
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path
