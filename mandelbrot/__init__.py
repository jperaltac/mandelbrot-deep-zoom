"""Public API for Mandelbrot rendering utilities."""

from .renderer import RenderParameters, RenderResult, SamplingMetadata, render_frame
from .generator import (
    ZoomPlanner,
    apply_zoom,
    compute_zoom_factors,
    pixel_to_complex,
    recenter_params,
    select_zoom_center,
)

__all__ = [
    "RenderParameters",
    "RenderResult",
    "SamplingMetadata",
    "ZoomPlanner",
    "apply_zoom",
    "compute_zoom_factors",
    "pixel_to_complex",
    "recenter_params",
    "render_frame",
    "select_zoom_center",
]
