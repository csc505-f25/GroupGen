"""
Optional GPU acceleration for GroupGen (distance matrix + K-Medoids hot paths).

Backends (first available when ``backend='auto'`` or ``'gpu'``):
  - **cuda** — NVIDIA via PyTorch
  - **directml** — AMD/Intel on Windows via torch-directml
  - **cpu** — NumPy / scikit-learn (default, always available)

Set ``GROUPGEN_BACKEND=gpu`` or pass ``backend='gpu'`` to the pipeline/API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

BackendName = Literal["auto", "cpu", "gpu", "cuda", "directml", "torch_cpu"]

_VALID_BACKENDS = frozenset({"auto", "cpu", "gpu", "cuda", "directml", "torch_cpu"})


@dataclass(frozen=True)
class ResolvedBackend:
    """Chosen execution backend after resolving user preference and hardware."""

    name: str  # "cpu" | "cuda" | "directml"
    label: str  # human-readable for logs/UI
    torch_device: object | None = None  # torch.device when GPU path is active


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _torch_directml_available() -> bool:
    try:
        import torch_directml  # type: ignore[import-untyped]

        _ = torch_directml.device()
        return True
    except ImportError:
        return False
    except Exception:
        return False


def list_available_backends() -> list[str]:
    """Backends that can be selected on this machine."""
    options = ["cpu"]
    if _torch_cuda_available():
        options.append("cuda")
    if _torch_directml_available():
        options.append("directml")
    return options


def resolve_backend(requested: str | None = None) -> ResolvedBackend:
    """
    Pick CPU or GPU backend.

    ``gpu`` / ``auto`` prefer CUDA, then DirectML, then CPU.
    """
    raw = (requested or os.environ.get("GROUPGEN_BACKEND", "cpu")).strip().lower()
    if raw not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown backend {requested!r}. Use one of: {', '.join(sorted(_VALID_BACKENDS))}."
        )

    if raw == "cpu":
        return ResolvedBackend(name="cpu", label="CPU (NumPy / scikit-learn)")

    if raw == "torch_cpu":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "torch_cpu backend requires PyTorch. Install: pip install torch"
            ) from exc
        return ResolvedBackend(
            name="torch_cpu",
            label="PyTorch on CPU (comparison only)",
            torch_device=torch.device("cpu"),
        )

    if raw == "cuda":
        if not _torch_cuda_available():
            raise RuntimeError(
                "CUDA backend requested but PyTorch CUDA is not available. "
                "Install: pip install torch --index-url https://download.pytorch.org/whl/cu124"
            )
        import torch

        return ResolvedBackend(
            name="cuda",
            label=f"CUDA ({torch.cuda.get_device_name(0)})",
            torch_device=torch.device("cuda"),
        )

    if raw == "directml":
        if not _torch_directml_available():
            raise RuntimeError(
                "DirectML backend requested but torch-directml is not installed. "
                "Install: pip install torch-directml"
            )
        import torch_directml  # type: ignore[import-untyped]

        return ResolvedBackend(
            name="directml",
            label="DirectML (AMD/Intel GPU)",
            torch_device=torch_directml.device(),
        )

    # auto or gpu — prefer CUDA, then DirectML, else CPU
    if _torch_cuda_available():
        import torch

        return ResolvedBackend(
            name="cuda",
            label=f"CUDA ({torch.cuda.get_device_name(0)})",
            torch_device=torch.device("cuda"),
        )
    if _torch_directml_available():
        import torch_directml  # type: ignore[import-untyped]

        return ResolvedBackend(
            name="directml",
            label="DirectML (AMD/Intel GPU)",
            torch_device=torch_directml.device(),
        )

    if raw == "gpu":
        raise RuntimeError(
            "GPU backend requested but no GPU runtime found. "
            "Install torch with CUDA (NVIDIA) or torch-directml (AMD/Intel on Windows)."
        )

    return ResolvedBackend(name="cpu", label="CPU (NumPy / scikit-learn)")
