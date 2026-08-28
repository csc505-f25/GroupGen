"""
Environment bootstrap for GG_analysis.ipynb.

Works in:
  - Cursor / VS Code with local Python (recommended — reads your docs/ folder)
  - Google Colab kernel in Cursor (clones repo; needs CRPY CSVs on disk or in git)
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/csc505-f25/GroupGen.git"
REPO_BRANCH = "Google_sheets_rawdata_intake"
COLAB_CLONE_ROOT = Path("/content/GroupGen")
SCRIPT_DIR = Path(__file__).resolve().parent


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def ensure_packages() -> None:
    required = ("pandas", "matplotlib", "seaborn", "numpy", "scipy")
    missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]
    if missing:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", *missing],
            stdout=subprocess.DEVNULL,
        )


def _count_crpy_csvs(root: Path) -> int:
    crpy = root / "Inclass_data" / "CRPY"
    return len(list(crpy.glob("*.csv"))) if crpy.is_dir() else 0


def _candidate_roots() -> list[Path]:
    cwd = Path.cwd()
    candidates = [
        SCRIPT_DIR,
        cwd,
        cwd / "docs",
        cwd.parent / "docs",
        SCRIPT_DIR.parent / "docs",
        SCRIPT_DIR.parent,
    ]
    if is_colab():
        candidates.extend(
            [
                COLAB_CLONE_ROOT / "docs",
                Path("/content/docs"),
                Path("/content"),
            ]
        )

    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    return ordered


def _clone_repo_on_colab() -> Path:
    docs = COLAB_CLONE_ROOT / "docs"
    if docs.is_dir() and (docs / "groupgen_analysis.py").exists():
        return docs

    COLAB_CLONE_ROOT.parent.mkdir(parents=True, exist_ok=True)
    if COLAB_CLONE_ROOT.is_dir():
        import shutil

        shutil.rmtree(COLAB_CLONE_ROOT)

    subprocess.check_call(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            REPO_BRANCH,
            REPO_URL,
            str(COLAB_CLONE_ROOT),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return docs


def resolve_docs_root() -> Path:
    for root in _candidate_roots():
        if (root / "groupgen_analysis.py").exists():
            return root

    if is_colab():
        return _clone_repo_on_colab()

    return SCRIPT_DIR


def init_session():
    """
    Notebook Setup entry point.

    Returns ``(ga, docs_root, info)`` where ``info`` is a small status dict.
    Raises ``FileNotFoundError`` with actionable text when CRPY data is missing.
    """
    ensure_packages()

    if is_colab():
        try:
            get_ipython().run_line_magic("matplotlib", "inline")  # type: ignore[name-defined]
        except Exception:
            pass

    docs_root = resolve_docs_root()
    if str(docs_root) not in sys.path:
        sys.path.insert(0, str(docs_root))

    import groupgen_analysis as ga

    importlib.reload(ga)
    ga.configure_study_paths(docs_root)

    crpy_dir = docs_root / "Inclass_data" / "CRPY"
    n_crpy = _count_crpy_csvs(docs_root)
    info = {
        "colab": is_colab(),
        "docs_root": docs_root,
        "crpy_dir": crpy_dir,
        "crpy_files": n_crpy,
        "ready": n_crpy >= 5,
    }

    print(f"Environment:  {'Colab' if info['colab'] else 'Local Python'}")
    print(f"Docs root:      {docs_root}")
    print(f"Output tables:  {ga.OUT_DIR}")
    print(f"Output figures: {ga.OUT_FIGURES}")
    print(f"CRPY CSVs:      {n_crpy}/5")

    if not info["ready"]:
        lines = [
            "CRPY study data not found.",
            f"Expected 5 CSV files in: {crpy_dir}",
        ]
        if info["colab"]:
            lines.extend(
                [
                    "",
                    "Colab runs in the cloud and cannot see your C: drive.",
                    "Fix: In Cursor, use Select Kernel -> your local venv (recommended),",
                    "     or push docs/Inclass_data/CRPY/*.csv to GitHub and re-run Setup.",
                ]
            )
        else:
            lines.append("Add the five CRPY survey exports to that folder, then re-run Setup.")
        raise FileNotFoundError("\n".join(lines))

    return ga, docs_root, info
