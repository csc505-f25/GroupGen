"""
HTTP helpers for consistent API error messages.

``format_error_detail`` turns FastAPI validation error lists into a single
readable string for the Next.js UI (avoids "[object Object]" on screen).
"""

from __future__ import annotations

from typing import Any


def format_error_detail(detail: Any) -> str:
    """Turn FastAPI validation errors or exceptions into one readable string."""
    if detail is None:
        return "An unknown error occurred."
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        # FastAPI 422 body: [{"loc": ["query", "group_size"], "msg": "..."}, ...]
        parts = []
        for item in detail:
            if isinstance(item, dict):
                loc = ".".join(str(x) for x in item.get("loc", ()))
                msg = item.get("msg", "")
                parts.append(f"{loc}: {msg}" if loc else str(msg))
            else:
                parts.append(str(item))
        return "\n".join(parts) if parts else "Validation failed."
    return str(detail)
