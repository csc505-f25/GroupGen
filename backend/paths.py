"""
Shared filesystem paths for the backend package.

All modules should resolve project-relative paths from ``BACKEND_DIR``
(the directory containing this file), not from the process cwd.
"""

from pathlib import Path

# .../GroupGen/backend
BACKEND_DIR = Path(__file__).resolve().parent

# .../GroupGen (repo root — parent of backend/)
PROJECT_ROOT = BACKEND_DIR.parent

DEFAULT_TEMPLATE_CSV = BACKEND_DIR / "data" / "templates" / "classroom_template.csv"
DEFAULT_GOOGLE_FORM_SAMPLE_CSV = (
    BACKEND_DIR / "data" / "templates" / "google_form_sample.csv"
)
DEFAULT_CLI_OUTPUT_DIR = BACKEND_DIR / "output" / "runs"
DEFAULT_EVAL_OUTPUT_DIR = BACKEND_DIR / "output_plots" / "runs"
