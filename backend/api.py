"""
GroupGen REST API — classroom study upload endpoint.

Flow:
  1. Accept multipart CSV + ``group_size`` query param
  2. ``prepare_for_grouping`` — auto-detects raw Google Forms exports via
     ``group_gen_intake`` and converts to clustering columns; HTTP 400 on bad data
  3. ``run_grouping_pipeline`` — HTTP 422 on invariant/size failures
  4. ``to_json_safe`` entire response (NumPy/pandas → JSON primitives)

The API is stateless: no files written server-side. See ``docs/ARCHITECTURE.md``.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np

from .data_loader import prepare_for_grouping
from .group_config import calculate_n_groups
from .http_util import format_error_detail
from .invariants import InvariantViolation
from .json_util import to_json_safe
from .pipeline import run_grouping_pipeline

app = FastAPI(title="GroupGen API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Study CSVs are small; cap prevents accidental huge uploads.
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": format_error_detail(exc.errors())},
    )


@app.get("/")
def health_check():
    return {"status": "GroupGen API is running"}


def _member_records(group_df: pd.DataFrame) -> list:
    """Student rows for API without internal Group_ID column (id lives on parent group)."""
    cols = [c for c in group_df.columns if c != "Group_ID"]
    return group_df[cols].replace({np.nan: None}).to_dict(orient="records")


@app.post("/generate-groups")
async def generate_groups(
    file: UploadFile = File(...),
    group_size: int = Query(5, ge=2, le=50, description="Target students per group"),
):
    """Upload a CSV (e.g. from Google Forms) and receive balanced groups."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv export.")

    contents = await file.read()  # raw bytes — parsed only inside prepare_for_grouping

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 5 MB).")
    if not contents.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # --- Ingest boundary: nothing below runs if CSV is invalid ---
    try:
        df = prepare_for_grouping(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    try:
        result = run_grouping_pipeline(df, group_size, verbose=False)
        labels = result.labels
        # 1-based Group_ID for teachers; clustering still used 0-based labels internally.
        df = df.copy()
        df["Group_ID"] = labels + 1

        response_groups = []
        unique_ids = sorted(df["Group_ID"].unique())

        for g_id in unique_ids:
            group_df = df[df["Group_ID"] == g_id]
            members = _member_records(group_df)
            gender_balance = to_json_safe(
                group_df["Gender"].value_counts().to_dict()
            )
            learning_styles = to_json_safe(
                group_df["Learning_Style"].mode().tolist()
            )
            stats = {
                "size": len(members),
                "avg_motivation": round(float(group_df["Motivation"].mean()), 2),
                "avg_self_esteem": round(float(group_df["Self_Esteem"].mean()), 2),
                "avg_work_ethic": round(float(group_df["Work_Ethic"].mean()), 2),
                "gender_balance": gender_balance,
                "learning_styles": learning_styles,
            }
            response_groups.append(
                {"id": int(g_id), "members": members, "stats": stats}
            )

        # Warnings mean groups are valid but fairness may need a manual look.
        status = "success" if not result.warnings else "success_with_warnings"

        # Whole tree sanitized so browser never sees np.int64 / nan / inf.
        return to_json_safe({
            "status": status,
            "total_students": len(df),
            "total_groups": len(unique_ids),
            "target_group_size": group_size,
            "configured_groups": calculate_n_groups(len(df), group_size),
            "group_size_summary": result.group_size_range,
            "warnings": result.warnings,
            "groups": response_groups,
        })

    except InvariantViolation as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grouping failed: {e}")
