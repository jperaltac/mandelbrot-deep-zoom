"""Rendering primitives for Mandelbrot frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorflow as tf

HORIZON = 4


@dataclass(frozen=True)
class RenderParameters:
    """Parameters that describe a single render of the Mandelbrot set."""

    x_res: int
    y_res: int
    x_center: float
    y_center: float
    x_width: float
    y_width: float
    max_iterations: int


@dataclass(frozen=True)
class SamplingMetadata:
    """Metadata describing the sampling grid for a rendered frame."""

    x_min: float
    y_min: float
    x_step: float
    y_step: float
    x_res: int
    y_res: int


@dataclass(frozen=True)
class RenderResult:
    """Container for the numerical results of a Mandelbrot render."""

    smooth: np.ndarray
    iterations: np.ndarray
    edges: np.ndarray
    metadata: SamplingMetadata


@tf.function
def _mandelbrot_step(zs: tf.Tensor, xs: tf.Tensor, ns: tf.Tensor, active: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Perform a single Mandelbrot iteration for points that have not diverged."""

    zs_new = zs * zs + xs
    zs = tf.where(active, zs_new, zs)
    ns = ns + tf.cast(active, tf.int32)
    az = tf.abs(zs)
    horizon = tf.cast(HORIZON, az.dtype)
    new_active = tf.logical_and(active, az < horizon)
    return zs, ns, new_active


@tf.function
def _mandelbrot_run(xs: tf.Tensor, zs: tf.Tensor, ns: tf.Tensor, max_iterations: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Iterate the Mandelbrot formula using a TensorFlow while loop."""

    max_iterations = tf.cast(max_iterations, tf.int32)
    i = tf.constant(0, dtype=tf.int32)
    active = tf.ones_like(ns, tf.bool)

    def cond(i: tf.Tensor, zs: tf.Tensor, ns: tf.Tensor, active: tf.Tensor) -> tf.Tensor:
        return tf.logical_and(tf.less(i, max_iterations), tf.reduce_any(active))

    def body(i: tf.Tensor, zs: tf.Tensor, ns: tf.Tensor, active: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        zs, ns, active = _mandelbrot_step(zs, xs, ns, active)
        return i + 1, zs, ns, active

    return tf.while_loop(cond, body, (i, zs, ns, active))


def _compute_metadata(params: RenderParameters) -> SamplingMetadata:
    x_res = max(int(params.x_res), 1)
    y_res = max(int(params.y_res), 1)

    x_width = np.float64(params.x_width)
    y_width = np.float64(params.y_width)
    x_center = np.float64(params.x_center)
    y_center = np.float64(params.y_center)

    x_min = x_center - x_width / 2.0
    x_max = x_center + x_width / 2.0
    y_min = y_center - y_width / 2.0
    y_max = y_center + y_width / 2.0

    x_step = np.float64((x_max - x_min) / (x_res - 1)) if x_res > 1 else np.float64(0.0)
    y_step = np.float64((y_max - y_min) / (y_res - 1)) if y_res > 1 else np.float64(0.0)

    return SamplingMetadata(
        x_min=float(x_min),
        y_min=float(y_min),
        x_step=float(x_step),
        y_step=float(y_step),
        x_res=x_res,
        y_res=y_res,
    )


def render_frame(params: RenderParameters, *, device: Optional[str] = None) -> RenderResult:
    """Render a Mandelbrot frame given the supplied parameters."""

    metadata = _compute_metadata(params)
    x_res = metadata.x_res
    y_res = metadata.y_res

    x = np.linspace(metadata.x_min, metadata.x_min + metadata.x_step * (x_res - 1), x_res, dtype=np.float64) if x_res > 1 else np.array([metadata.x_min], dtype=np.float64)
    y = np.linspace(metadata.y_min, metadata.y_min + metadata.y_step * (y_res - 1), y_res, dtype=np.float64) if y_res > 1 else np.array([metadata.y_min], dtype=np.float64)

    max_iterations = tf.constant(params.max_iterations, dtype=tf.int32)

    with tf.device(device if device is not None else "/CPU:0"):
        x_tf = tf.convert_to_tensor(x, dtype=tf.float64)
        y_tf = tf.convert_to_tensor(y, dtype=tf.float64)
        X, Y = tf.meshgrid(x_tf, y_tf)
        Z = tf.complex(X, Y)
        xs = tf.identity(Z)
        zs = tf.identity(xs)
        ns = tf.zeros_like(xs, tf.int32)

        _, zs, ns, _ = _mandelbrot_run(xs, zs, ns, max_iterations)

        az = tf.abs(zs)
        eps = tf.constant(1e-12, dtype=az.dtype)
        az_safe = tf.maximum(az, tf.constant(1.0, dtype=az.dtype) + eps)
        log_az = tf.math.log(az_safe)
        log_log_az = tf.math.log(tf.maximum(log_az, eps))
        ns_float = tf.cast(ns, tf.float64)
        log2 = tf.math.log(tf.constant(2.0, dtype=az.dtype))
        smooth_escape = ns_float + tf.constant(1.0, dtype=ns_float.dtype) - log_log_az / log2
        escaped = tf.less(ns, max_iterations)
        smooth = tf.where(escaped, smooth_escape, tf.cast(max_iterations, tf.float64))

        max_tensor = tf.cast(max_iterations, ns.dtype)
        mask = tf.cast(
            tf.clip_by_value(tf.fill(tf.shape(ns), max_tensor) - ns, 0, 1),
            tf.bool,
        )
        edges = tf.math.logical_xor(tf.roll(mask, 1, axis=0), mask)

    return RenderResult(
        smooth=smooth.numpy(),
        iterations=ns.numpy(),
        edges=edges.numpy(),
        metadata=metadata,
    )
