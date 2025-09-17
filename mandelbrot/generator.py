"""Utilities for managing Mandelbrot zoom sequences."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .renderer import RenderParameters, RenderResult, SamplingMetadata


@dataclass(frozen=True)
class ZoomPlanner:
    """Maintain the parameter updates for a Mandelbrot zoom sequence."""

    lock_aspect: bool
    aspect: float

    def enforce_aspect(self, params: RenderParameters) -> RenderParameters:
        if not self.lock_aspect:
            return params
        new_y_width = np.float64(params.x_width) * np.float64(self.aspect)
        return replace(params, y_width=float(new_y_width))

    def initialize_focus(self, params: RenderParameters, result: RenderResult) -> RenderParameters:
        focus_pixel = select_zoom_center(result.edges)
        return recenter_params(params, focus_pixel, result.metadata)

    def update_after_frame(self, params: RenderParameters, result: RenderResult, zoom_factor: float) -> RenderParameters:
        focus_pixel = select_zoom_center(result.edges)
        params = recenter_params(params, focus_pixel, result.metadata)
        return apply_zoom(params, zoom_factor, self.lock_aspect, self.aspect)


def compute_zoom_factors(frames: int, zoom_factor: float, *, final_zoom: float | None, easing: str) -> np.ndarray:
    """Compute per-frame zoom multipliers for the animation."""

    if frames <= 0:
        return np.array([], dtype=np.float64)

    if final_zoom is not None and final_zoom > 0:
        total_frames = max(1, frames)
        log_target = np.log(final_zoom)
        easing_mode = easing.lower()

        def ease_in_out(t: float) -> float:
            return 3 * t ** 2 - 2 * t ** 3

        ease = (lambda u: u) if easing_mode == "linear" else ease_in_out
        if total_frames == 1:
            alphas = np.array([1.0], dtype=np.float64)
        else:
            alphas = np.array([ease(i / (total_frames - 1)) for i in range(total_frames)], dtype=np.float64)
        alphas = np.clip(alphas, 0.0, 1.0)
        increments = np.diff(np.concatenate(([0.0], alphas)))
        per_frame_logs = increments * log_target
        if frames < total_frames:
            per_frame_logs = per_frame_logs[:frames]
        return np.exp(per_frame_logs)

    return np.full(frames, np.float64(zoom_factor), dtype=np.float64)


def select_zoom_center(edges: np.ndarray) -> np.ndarray:
    """Select a deterministic focus pixel near the center of the edge map."""

    if edges.size == 0:
        return np.array([edges.shape[0] // 2, edges.shape[1] // 2], dtype=np.int64)

    height, width = edges.shape
    center_row = height // 2
    center_col = width // 2
    max_radius = max(height, width)

    for radius in range(max_radius):
        row_start = max(center_row - radius, 0)
        row_end = min(center_row + radius + 1, height)
        col_start = max(center_col - radius, 0)
        col_end = min(center_col + radius + 1, width)
        region = edges[row_start:row_end, col_start:col_end]
        if np.any(region):
            indices = np.argwhere(region)
            indices[:, 0] += row_start
            indices[:, 1] += col_start
            return _select_zoom_center(indices, edges.shape)

    edge_indices = np.argwhere(edges)
    if edge_indices.size:
        return _select_zoom_center(edge_indices, edges.shape)
    return np.array([center_row, center_col], dtype=np.int64)


def _select_zoom_center(edge_indices: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if edge_indices.size == 0:
        return np.array([shape[0] // 2, shape[1] // 2], dtype=np.int64)

    center = np.array([(shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0], dtype=np.float64)
    indices = edge_indices.astype(np.float64, copy=False)
    distances = np.sum((indices - center) ** 2, axis=1)
    best_idx = int(np.argmin(distances))
    return edge_indices[best_idx]


def recenter_params(params: RenderParameters, pixel: np.ndarray, metadata: SamplingMetadata) -> RenderParameters:
    x_center, y_center = pixel_to_complex(metadata, pixel[0], pixel[1])
    return replace(params, x_center=float(x_center), y_center=float(y_center))


def apply_zoom(params: RenderParameters, zoom_factor: float, lock_aspect: bool, aspect: float) -> RenderParameters:
    x_width = np.float64(params.x_width) * np.float64(zoom_factor)
    if lock_aspect:
        y_width = np.float64(x_width) * np.float64(aspect)
    else:
        y_width = np.float64(params.y_width) * np.float64(zoom_factor)
    return replace(params, x_width=float(x_width), y_width=float(y_width))


def pixel_to_complex(metadata: SamplingMetadata, row: int, col: int) -> tuple[np.float64, np.float64]:
    if metadata.x_res > 1:
        x = np.float64(metadata.x_min) + np.float64(col) * np.float64(metadata.x_step)
    else:
        x = np.float64(metadata.x_min)
    if metadata.y_res > 1:
        y = np.float64(metadata.y_min) + np.float64(row) * np.float64(metadata.y_step)
    else:
        y = np.float64(metadata.y_min)
    return np.float64(x), np.float64(y)
