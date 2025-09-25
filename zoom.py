import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_VERBOSE_FLAGS = {"--verbose", "-v"}
_cli_verbose = any(arg in _VERBOSE_FLAGS for arg in sys.argv[1:])
_env_log_level = os.environ.get("TF_CPP_MIN_LOG_LEVEL")
_suppress_messages = (not _cli_verbose) and _env_log_level != "0"

if _suppress_messages and _env_log_level is None:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if _suppress_messages:
    warnings.filterwarnings(
        "ignore",
        message=r"Protobuf gencode version .* is exactly one major version older than the runtime version .*",
        category=UserWarning,
        module="google.protobuf",
    )

VERBOSE = _cli_verbose


def log(message, *args, **kwargs):
    if VERBOSE:
        print(message, *args, **kwargs)


# Import libraries for simulation
import tensorflow as tf
import numpy as np

if _suppress_messages:
    try:
        tf.get_logger().setLevel("ERROR")
        for handler in tf.get_logger().handlers:
            handler.setLevel("ERROR")
    except Exception:
        pass

# Imports for visualization
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import imageio

from mandelbrot import (
    RenderParameters,
    SamplingMetadata,
    ZoomPlanner,
    compute_zoom_factors,
    render_frame,
)

try:
    from matplotlib import colormaps as _mpl_colormaps
except ImportError:  # Matplotlib < 3.5
    from matplotlib import cm as _mpl_colormaps  # type: ignore


def get_colormap(name):
    return _mpl_colormaps.get_cmap(name)


log("TensorFlow version: %s" % tf.__version__)

# Configure TensorFlow to use the GPU when available. This also allows the code
# to run on systems without a GPU, such as the execution environment used for
# testing. On a V100 or any other CUDA-capable card, TensorFlow will place the
# computation on the first visible GPU.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        DEVICE = '/GPU:0'
        log("GPU found, using %s" % gpus[0].name)
    except RuntimeError as e:
        if VERBOSE:
            print(e)
        DEVICE = '/CPU:0'
else:
    DEVICE = '/CPU:0'
    log("No GPU found, using CPU")

from argparse import ArgumentParser


@dataclass
class OutputConfig:
    modes: tuple[str, ...]
    gif_path: Path | None
    image_path: Path | None
    frame_dir: Path | None
    save_color_frames: bool
    save_mono_frames: bool
    keep_frames: bool
    image_format: str


def build_parser():
    parser = ArgumentParser()

    parser.add_argument('--max-iterations', type=int,
                        dest='max_iterations', help='maximum number of times to iterate logistic mapping',
                        metavar='MAX_ITERATIONS', default=2000)

    parser.add_argument('--x-res', type=int,
                        dest='x_res', help='resolution of samples along the x-axis',
                        metavar='X_RES', default=512)

    parser.add_argument('--y-res', type=int,
                        dest='y_res', help='resolution of samples along the y-axis',
                        metavar='Y_RES', default=512)

    parser.add_argument('--x-center', type=float,
                        dest='x_center', help='x coordinate in the complex plane to start the zoom at',
                        metavar='X_CENTER', default=-0.75)

    parser.add_argument('--x-width', type=float,
                        dest='x_width', help='starting width of the sample window in the complex plane',
                        metavar='X_WIDTH', default=2.5)

    parser.add_argument('--y-center', type=float,
                        dest='y_center', help='y coordinate in the complex plane to start the zoom at',
                        metavar='Y_CENTER', default=0)

    parser.add_argument('--y-width', type=float,
                        dest='y_width', help='starting height of the sample window in the complex plane',
                        metavar='Y_WIDTH', default=2.5)

    parser.add_argument('--lock-aspect', action='store_true',
                        help='Maintains y_width = x_width * (y_res/x_res) to avoid stretching when zooming.')

    parser.add_argument('--zoom-factor', type=float,
                        dest='zoom_factor', help='the factor by which to multiply the window size each frame. Choose < 1 for zoom in, >1 for zoom out',
                        metavar='ZOOM_FACTOR', default=0.8)

    parser.add_argument('--final-zoom', type=float, default=None,
                        help='Overall scale applied by the last frame (e.g., 1e-4 narrows the window by 10000×). If set, overrides --zoom-factor.')

    parser.add_argument('--easing', type=str, default='ease',
                        help='Temporal curve used for variable zoom: "linear" or "ease" for smooth ease-in-out.')

    parser.add_argument('--frames', type=int,
                        dest='frames', help='number of frames to generate',
                        metavar='FRAMES', default=100)

    parser.add_argument('--mode', dest='modes', action='append', metavar='MODE',
                        help='Output modes to generate. May be repeated. Choices: gif, image, frames, mono.')

    parser.add_argument('--output', dest='output', type=str,
                        help='Destination for single-file outputs (gif/image) or container directory when both are requested.')

    parser.add_argument('--frame-dir', dest='frame_dir', type=str,
                        help='Directory in which to store frame sequences (frames/mono or temporary GIF frames).')

    parser.add_argument('--keep-frames', dest='keep_frames', action='store_true',
                        help='When generating a GIF, keep the individual frames in frame-dir in addition to the GIF file.')

    parser.add_argument('--save-frames', dest='legacy_save_frames', action='store_true',
                        help='[DEPRECATED] Equivalent to --mode frames. May not be combined with --mode.')

    parser.add_argument('--save-mono', dest='legacy_save_mono', action='store_true',
                        help='[DEPRECATED] Equivalent to --mode mono. May not be combined with --mode.')

    parser.add_argument('--colormap', type=str,
                        dest='colormap', help='matplotlib colormap to colorize the fractal (e.g. "viridis", "inferno")',
                        metavar='COLORMAP', default='twilight_shifted')

    parser.add_argument('--format', type=str,
                        dest='format', help='file format for image-based outputs. Can be any extension supported by Pillow. Default: "png".',
                        metavar='FORMAT', default='png')

    parser.add_argument('--frames-path', type=str,
                        dest='legacy_frames_path', help='[DEPRECATED] Alias for --frame-dir.',
                        metavar='FRAMES_PATH')

    parser.add_argument('--show-edges', help='render the edge detection beside',
                        dest='show_edges', action="store_true")
    parser.add_argument('--show-coordinates', help='overlay axis bounds and origin marker on the final image',
                        dest='show_coordinates', action='store_true')

    parser.add_argument('--normalize', choices=['outside', 'all'], default='outside',
                        help='Normalization strategy: "outside" uses only escaping points; "all" uses every sample.')
    parser.add_argument('--gamma', type=float, default=0.85, help='Gamma correction for tone mapping.')
    parser.add_argument('--clip-low', type=float, default=0.5, help='Lower percentile for normalization clipping.')
    parser.add_argument('--clip-high', type=float, default=99.5, help='Upper percentile for normalization clipping.')
    parser.add_argument('--invert', action='store_true', help='Invert the selected colormap.')
    parser.add_argument('--inside-color', type=str, default='#000000', help='Hex color for points inside the Mandelbrot set.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose logging, including TensorFlow and hardware diagnostics.')

    return parser


def _normalize_path(path_str: str) -> str:
    return os.path.normpath(os.path.abspath(os.path.expanduser(path_str)))


def resolve_output_config(opt, parser: ArgumentParser) -> OutputConfig:
    valid_modes = {"gif", "image", "frames", "mono"}
    modes: list[str] = []
    explicit_modes = opt.modes or []
    if explicit_modes:
        modes.extend(explicit_modes)

    if getattr(opt, "legacy_save_frames", False):
        if explicit_modes:
            parser.error("--save-frames (deprecated) cannot be combined with --mode.")
        warnings.warn("--save-frames is deprecated; use --mode frames instead.", DeprecationWarning, stacklevel=2)
        modes.append("frames")

    if getattr(opt, "legacy_save_mono", False):
        if explicit_modes:
            parser.error("--save-mono (deprecated) cannot be combined with --mode.")
        warnings.warn("--save-mono is deprecated; use --mode mono instead.", DeprecationWarning, stacklevel=2)
        modes.append("mono")

    if not modes:
        modes = ["gif"]

    normalized_modes: list[str] = []
    for mode in modes:
        if mode not in valid_modes:
            parser.error(f"Unknown output mode '{mode}'. Valid choices: {', '.join(sorted(valid_modes))}.")
        if mode not in normalized_modes:
            normalized_modes.append(mode)

    modes_tuple = tuple(normalized_modes)
    modes_set = set(modes_tuple)

    keep_frames = bool(getattr(opt, "keep_frames", False))
    if keep_frames and "gif" not in modes_set:
        parser.error("--keep-frames requires the gif mode.")

    frame_dir_explicit = getattr(opt, "frame_dir", None)
    legacy_frame_dir = getattr(opt, "legacy_frames_path", None)
    if frame_dir_explicit and legacy_frame_dir:
        normalized_new = _normalize_path(frame_dir_explicit)
        normalized_legacy = _normalize_path(legacy_frame_dir)
        if normalized_new != normalized_legacy:
            parser.error("--frame-dir and --frames-path refer to different locations.")

    frame_dir_value = frame_dir_explicit or legacy_frame_dir

    needs_frame_dir = bool({"frames", "mono"} & modes_set or ("gif" in modes_set and keep_frames))
    frame_dir_path: Path | None = None
    if needs_frame_dir:
        frame_dir_value = frame_dir_value or "./frames"
        frame_dir_path = Path(frame_dir_value).expanduser().resolve()
    else:
        if frame_dir_value is not None:
            parser.error("--frame-dir is only valid with frames/mono modes or --keep-frames.")

    image_format = (getattr(opt, "format", "png") or "png").lower().lstrip(".")
    if not image_format:
        image_format = "png"

    file_modes = [mode for mode in modes_tuple if mode in {"gif", "image"}]
    output_arg = getattr(opt, "output", None)
    gif_path: Path | None = None
    image_path: Path | None = None

    if not file_modes:
        if output_arg:
            parser.error("--output is only valid when gif or image modes are requested.")
    elif len(file_modes) == 1:
        mode = file_modes[0]
        if output_arg:
            output_path = Path(output_arg).expanduser()
            if str(output_arg).endswith(tuple(filter(None, {os.sep, os.altsep}))) or str(output_arg).endswith("/"):
                parser.error("--output must be a file path when a single file-based mode is selected.")
            if output_path.exists() and output_path.is_dir():
                parser.error("--output must point to a file, not a directory, when a single file mode is active.")
            if mode == "gif":
                if output_path.suffix:
                    if output_path.suffix.lower() != ".gif":
                        parser.error("GIF outputs must end with .gif.")
                else:
                    output_path = output_path.with_suffix(".gif")
                gif_path = output_path.expanduser().resolve()
            else:
                suffix = output_path.suffix
                expected_suffix = f".{image_format}"
                if suffix:
                    if suffix.lower() != expected_suffix.lower():
                        parser.error(f"--output extension {suffix} does not match --format {image_format}.")
                else:
                    output_path = output_path.with_suffix(expected_suffix)
                image_path = output_path.expanduser().resolve()
        else:
            if mode == "gif":
                gif_path = Path("movie.gif").expanduser().resolve()
            else:
                image_path = Path(f"frame_final.{image_format}").expanduser().resolve()
    else:
        base_dir = Path(output_arg).expanduser() if output_arg else Path.cwd()
        if base_dir.exists():
            if not base_dir.is_dir():
                parser.error("--output must be a directory when both gif and image modes are active.")
        else:
            base_dir.mkdir(parents=True, exist_ok=True)
        gif_path = (base_dir / "movie.gif").expanduser().resolve()
        image_path = (base_dir / f"frame_final.{image_format}").expanduser().resolve()

    if gif_path is None and "gif" in modes_set:
        gif_path = Path("movie.gif").expanduser().resolve()
    if image_path is None and "image" in modes_set:
        image_path = Path(f"frame_final.{image_format}").expanduser().resolve()

    save_color_frames = "frames" in modes_set or ("gif" in modes_set and keep_frames)
    save_mono_frames = "mono" in modes_set

    return OutputConfig(
        modes=modes_tuple,
        gif_path=gif_path,
        image_path=image_path,
        frame_dir=frame_dir_path,
        save_color_frames=save_color_frames,
        save_mono_frames=save_mono_frames,
        keep_frames=keep_frames,
        image_format=image_format,
    )


def _pil_format_name(ext: str) -> str:
    upper = ext.upper()
    if upper == "JPG":
        return "JPEG"
    if upper == "TIF":
        return "TIFF"
    return upper


def write_single_image(image: PIL.Image.Image, output_path: Path, image_format: str) -> None:
    """Write a single image to ``output_path`` using the provided format."""

    pil_format = _pil_format_name(image_format)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output_path), format=pil_format)


def write_frame_sequence(
    image: PIL.Image.Image,
    frame_dir: Path,
    index: int,
    digits: int,
    image_format: str,
    prefix: str,
) -> Path:
    """Persist a frame in a numbered sequence inside ``frame_dir``."""

    pil_format = _pil_format_name(image_format)
    frame_path = frame_dir / f"{prefix}{index:0{digits}d}.{image_format}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    image.save(str(frame_path), format=pil_format)
    return frame_path


def write_gif(writer: Any, frame_array: np.ndarray) -> None:
    """Append ``frame_array`` to an active GIF writer."""

    writer.append_data(frame_array)


_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
)


def _load_annotation_font(image: PIL.Image.Image, scale: float = 1.0) -> PIL.ImageFont.ImageFont:
    base = max(min(image.size), 1)
    target_size = max(12, int(round(base * 0.028 * scale)))
    for path in _FONT_CANDIDATES:
        font_path = Path(path)
        if font_path.exists():
            try:
                return PIL.ImageFont.truetype(str(font_path), target_size)
            except OSError:
                continue
    return PIL.ImageFont.load_default()


def _create_vertical_gradient(
    size: tuple[int, int],
    top_color: tuple[int, int, int, int],
    bottom_color: tuple[int, int, int, int],
) -> PIL.Image.Image:
    width, height = size
    if height <= 0 or width <= 0:
        return PIL.Image.new("RGBA", (max(width, 1), max(height, 1)), (0, 0, 0, 0))

    gradient = PIL.Image.new("RGBA", (width, height))
    if height == 1:
        gradient.paste(top_color, [0, 0, width, height])
        return gradient

    top = tuple(top_color)
    bottom = tuple(bottom_color)
    for y in range(height):
        ratio = y / (height - 1)
        color = tuple(
            int(round(top[channel] + (bottom[channel] - top[channel]) * ratio))
            for channel in range(4)
        )
        gradient.paste(color, [0, y, width, y + 1])
    return gradient


def _gradient_panel(
    size: tuple[int, int],
    radius: int,
    top_color: tuple[int, int, int, int],
    bottom_color: tuple[int, int, int, int],
) -> PIL.Image.Image:
    gradient = _create_vertical_gradient(size, top_color, bottom_color)
    if radius <= 0:
        return gradient

    mask = PIL.Image.new("L", size, 0)
    mask_draw = PIL.ImageDraw.Draw(mask)
    corner_radius = min(radius, min(size) // 2)
    rounded_rect = getattr(mask_draw, "rounded_rectangle", None)
    if rounded_rect is not None:
        rounded_rect([(0, 0), (size[0] - 1, size[1] - 1)], radius=corner_radius, fill=255)
    else:
        mask_draw.rectangle([(0, 0), (size[0] - 1, size[1] - 1)], fill=255)
    transparent = PIL.Image.new("RGBA", size, (0, 0, 0, 0))
    return PIL.Image.composite(gradient, transparent, mask)


def _draw_text_with_shadow(
    draw: PIL.ImageDraw.ImageDraw,
    position: tuple[float, float],
    text: str,
    font: PIL.ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    *,
    shadow_fill: tuple[int, int, int, int] = (0, 0, 0, 160),
    shadow_offset: tuple[int, int] = (2, 2),
    spacing: int = 0,
) -> None:
    shadow_x = position[0] + shadow_offset[0]
    shadow_y = position[1] + shadow_offset[1]
    multiline_text = getattr(draw, "multiline_text", None)
    use_multiline = "\n" in text and multiline_text is not None

    if use_multiline:
        multiline_text((shadow_x, shadow_y), text, font=font, fill=shadow_fill, spacing=spacing)
        multiline_text(position, text, font=font, fill=fill, spacing=spacing)
    else:
        draw.text((shadow_x, shadow_y), text, font=font, fill=shadow_fill)
        if "\n" in text and multiline_text is not None:
            multiline_text(position, text, font=font, fill=fill, spacing=spacing)
        else:
            draw.text(position, text, font=font, fill=fill)


def annotate_with_coordinates(
    image: PIL.Image.Image,
    metadata: SamplingMetadata,
    params: RenderParameters,
) -> PIL.Image.Image:
    """Overlay axis ranges and the complex origin on ``image``."""

    if image.mode != "RGBA":
        image = image.convert("RGBA")

    draw = PIL.ImageDraw.Draw(image, "RGBA")
    panel_font = _load_annotation_font(image, scale=1.0)
    label_font = _load_annotation_font(image, scale=0.85)
    panel_font_size = getattr(panel_font, "size", 14)
    label_font_size = getattr(label_font, "size", 12)

    x_max = metadata.x_min + metadata.x_step * max(metadata.x_res - 1, 0)
    y_max = metadata.y_min + metadata.y_step * max(metadata.y_res - 1, 0)

    lines = [
        f"X: [{metadata.x_min:.6g}, {x_max:.6g}]",
        f"Y: [{metadata.y_min:.6g}, {y_max:.6g}]",
        f"Centro: ({params.x_center:.6g}, {params.y_center:.6g})",
    ]

    padding = max(8, int(round(panel_font_size * 0.6)))
    line_spacing = max(4, int(round(panel_font_size * 0.35)))
    text_width = 0
    line_heights: list[int] = []

    bbox_fn = getattr(draw, "textbbox", None)
    for line in lines:
        if bbox_fn is not None:
            bbox = bbox_fn((0, 0), line, font=panel_font)
            width = int(round(bbox[2] - bbox[0]))
            height = int(round(bbox[3] - bbox[1]))
        else:
            width, height = draw.textsize(line, font=panel_font)
        text_width = max(text_width, width)
        line_heights.append(height)

    total_text_height = sum(line_heights)
    if lines:
        total_text_height += line_spacing * (len(lines) - 1)

    box_width = text_width + padding * 2
    box_height = total_text_height + padding * 2
    box_left = 12
    box_top = 12
    box_right = box_left + box_width
    box_bottom = box_top + box_height

    panel_radius = max(10, int(round(min(box_width, box_height) * 0.18)))
    panel_overlay = _gradient_panel(
        (box_width, box_height),
        panel_radius,
        (18, 22, 40, 210),
        (10, 12, 24, 150),
    )
    image.paste(panel_overlay, (box_left, box_top), panel_overlay)

    rounded_rect = getattr(draw, "rounded_rectangle", None)
    outline_width = max(1, int(round(panel_font_size * 0.08)))
    outline_color = (255, 255, 255, 45)
    panel_outline_radius = min(panel_radius, min(box_width, box_height) // 2)
    if rounded_rect is not None:
        rounded_rect(
            [(box_left, box_top), (box_right, box_bottom)],
            radius=panel_outline_radius,
            outline=outline_color,
            width=outline_width,
        )
    else:
        draw.rectangle(
            [(box_left, box_top), (box_right, box_bottom)],
            outline=outline_color,
            width=outline_width,
        )

    text_y = box_top + padding
    shadow_offset = (
        max(1, int(round(panel_font_size * 0.1))),
        max(1, int(round(panel_font_size * 0.1))),
    )
    for idx, (line, height) in enumerate(zip(lines, line_heights)):
        _draw_text_with_shadow(
            draw,
            (box_left + padding, text_y),
            line,
            panel_font,
            (240, 244, 255, 255),
            shadow_fill=(0, 0, 0, 170),
            shadow_offset=shadow_offset,
        )
        increment = height + (line_spacing if idx < len(lines) - 1 else 0)
        text_y += increment

    x_bounds = (metadata.x_min, x_max)
    y_bounds = (metadata.y_min, y_max)
    zero_in_x = min(x_bounds) <= 0.0 <= max(x_bounds)
    zero_in_y = min(y_bounds) <= 0.0 <= max(y_bounds)

    if zero_in_x and zero_in_y and metadata.x_res > 0 and metadata.y_res > 0:
        if metadata.x_step != 0.0 and metadata.y_step != 0.0:
            origin_col = int(round((0.0 - metadata.x_min) / metadata.x_step))
            origin_row = int(round((0.0 - metadata.y_min) / metadata.y_step))
            if 0 <= origin_col < metadata.x_res and 0 <= origin_row < metadata.y_res:
                radius = max(3, int(round(min(metadata.x_res, metadata.y_res) * 0.005)))
                ellipse_bbox = [
                    (origin_col - radius, origin_row - radius),
                    (origin_col + radius, origin_row + radius),
                ]
                shadow_radius = radius + max(1, radius // 3)
                shadow_bbox = [
                    (origin_col - shadow_radius, origin_row - shadow_radius),
                    (origin_col + shadow_radius, origin_row + shadow_radius),
                ]
                draw.ellipse(shadow_bbox, fill=(0, 0, 0, 120))
                draw.ellipse(
                    ellipse_bbox,
                    fill=(255, 255, 255, 235),
                    outline=(0, 0, 0, 180),
                    width=max(1, radius // 2),
                )

    region_width = min(image.width, metadata.x_res) if metadata.x_res > 0 else image.width
    region_height = min(image.height, metadata.y_res) if metadata.y_res > 0 else image.height

    if region_width > 0 and region_height > 0:
        label_margin = max(10, int(round(label_font_size * 1.1)))
        label_padding = max(4, int(round(label_font_size * 0.45)))
        label_spacing = max(2, int(round(label_font_size * 0.3)))

        multiline_bbox = getattr(draw, "multiline_textbbox", None)
        multiline_size = getattr(draw, "multiline_textsize", None)

        def measure(text: str) -> tuple[int, int]:
            if multiline_bbox is not None:
                bbox = multiline_bbox((0, 0), text, font=label_font, spacing=label_spacing)
                width = int(round(bbox[2] - bbox[0]))
                height = int(round(bbox[3] - bbox[1]))
            elif multiline_size is not None:
                width, height = multiline_size(text, font=label_font, spacing=label_spacing)
            else:
                width, height = draw.textsize(text, font=label_font)
            return width, height

        def clamp(value: float, low: float, high: float) -> float:
            return max(low, min(value, high))

        top_label_y = max(label_margin, box_bottom + label_margin)

        def compute_y(height: int, align: str) -> int:
            if align == "top":
                y = top_label_y
            elif align == "bottom":
                y = region_height - height - label_margin
            else:
                y = (region_height - height) / 2.0
            max_y = region_height - height - label_margin
            if max_y < label_margin:
                return int(round(clamp(y, 0, max(region_height - height, 0))))
            return int(round(clamp(y, label_margin, max_y)))

        def compute_x(width: int, align: str) -> int:
            if align == "left":
                x = label_margin
            elif align == "right":
                x = region_width - width - label_margin
            else:
                x = (region_width - width) / 2.0
            max_x = region_width - width - label_margin
            if max_x < label_margin:
                return int(round(clamp(x, 0, max(region_width - width, 0))))
            return int(round(clamp(x, label_margin, max_x)))

        def format_coord(x: float, y: float) -> str:
            return f"x={x:.6g}\ny={y:.6g}"

        label_definitions: list[tuple[float, float, str, str]] = [
            (metadata.x_min, metadata.y_min, "left", "top"),
            (params.x_center, metadata.y_min, "center", "top"),
            (x_max, metadata.y_min, "right", "top"),
            (metadata.x_min, params.y_center, "left", "middle"),
            (params.x_center, params.y_center, "center", "middle"),
            (x_max, params.y_center, "right", "middle"),
            (metadata.x_min, y_max, "left", "bottom"),
            (params.x_center, y_max, "center", "bottom"),
            (x_max, y_max, "right", "bottom"),
        ]

        for x_value, y_value, align_x, align_y in label_definitions:
            text = format_coord(x_value, y_value)
            text_width, text_height = measure(text)
            x = compute_x(text_width, align_x)
            y = compute_y(text_height, align_y)
            rect_coords = [
                (x - label_padding, y - label_padding),
                (x + text_width + label_padding, y + text_height + label_padding),
            ]
            bg_width = text_width + label_padding * 2
            bg_height = text_height + label_padding * 2
            label_radius = max(6, int(round(label_font_size * 0.75)))
            label_overlay = _gradient_panel(
                (bg_width, bg_height),
                label_radius,
                (32, 38, 70, 200),
                (18, 22, 48, 150),
            )
            image.paste(label_overlay, (x - label_padding, y - label_padding), label_overlay)

            if rounded_rect is not None:
                label_outline_radius = min(label_radius, min(bg_width, bg_height) // 2)
                rounded_rect(
                    rect_coords,
                    radius=label_outline_radius,
                    outline=(255, 255, 255, 35),
                    width=max(1, int(round(label_font_size * 0.08))),
                )
            else:
                draw.rectangle(
                    rect_coords,
                    outline=(255, 255, 255, 35),
                    width=max(1, int(round(label_font_size * 0.08))),
                )

            label_shadow_offset = (
                max(1, int(round(label_font_size * 0.08))),
                max(1, int(round(label_font_size * 0.08))),
            )
            _draw_text_with_shadow(
                draw,
                (x, y),
                text,
                label_font,
                (235, 240, 255, 255),
                shadow_fill=(0, 0, 0, 160),
                shadow_offset=label_shadow_offset,
                spacing=label_spacing,
            )

    return image


@dataclass
class OutputWriters:
    config: OutputConfig
    frame_digits: int
    image_format: str
    show_edges: bool

    def __post_init__(self) -> None:
        self._needs_mono_frames = bool(self.config.save_mono_frames and self.config.frame_dir is not None)
        self._needs_color_frames = bool(self.config.save_color_frames and self.config.frame_dir is not None)
        self._needs_final_image = bool("image" in self.config.modes and self.config.image_path is not None)
        self._gif_writer = None
        if "gif" in self.config.modes and self.config.gif_path is not None:
            self._gif_writer = imageio.get_writer(str(self.config.gif_path), mode='I', duration=0.1, loop=0)

    def requires_color_array(self, frame_index: int, total_frames: int) -> bool:
        if self._needs_color_frames or self._gif_writer is not None:
            return True
        if self.show_edges and self._needs_final_image and total_frames:
            return frame_index == total_frames - 1
        return self._needs_final_image and total_frames and frame_index == total_frames - 1

    def requires_color_image(self, frame_index: int, total_frames: int) -> bool:
        if self._needs_color_frames:
            return True
        return self._needs_final_image and total_frames and frame_index == total_frames - 1

    def requires_mono_image(self) -> bool:
        return self._needs_mono_frames

    def should_store_final_image(self, frame_index: int, total_frames: int) -> bool:
        return self._needs_final_image and total_frames and frame_index == total_frames - 1

    def write_color_outputs(
        self,
        frame_index: int,
        frame_array: np.ndarray,
        color_image: PIL.Image.Image | None,
    ) -> None:
        if self._gif_writer is not None:
            write_gif(self._gif_writer, frame_array)
        if self._needs_color_frames and color_image is not None and self.config.frame_dir is not None:
            write_frame_sequence(
                color_image,
                self.config.frame_dir,
                frame_index,
                self.frame_digits,
                self.image_format,
                "frame",
            )

    def write_mono_output(self, frame_index: int, mono_image: PIL.Image.Image) -> None:
        if self._needs_mono_frames and self.config.frame_dir is not None:
            write_frame_sequence(
                mono_image,
                self.config.frame_dir,
                frame_index,
                self.frame_digits,
                self.image_format,
                "mono",
            )

    def finalize(self, final_image: PIL.Image.Image | None) -> None:
        if self._needs_final_image and final_image is not None and self.config.image_path is not None:
            write_single_image(final_image, self.config.image_path, self.image_format)

    def close(self) -> None:
        if self._gif_writer is not None:
            self._gif_writer.close()
            self._gif_writer = None




def main():
    parser = build_parser()
    opt = parser.parse_args()

    output_config = resolve_output_config(opt, parser)

    global VERBOSE
    VERBOSE = bool(opt.verbose)

    initial_params = RenderParameters(
        x_res=opt.x_res,
        y_res=opt.y_res,
        x_center=opt.x_center,
        y_center=opt.y_center,
        x_width=opt.x_width,
        y_width=opt.y_width,
        max_iterations=opt.max_iterations,
    )

    aspect = np.float64(opt.y_res / opt.x_res) if opt.x_res != 0 else np.float64(1.0)
    planner = ZoomPlanner(lock_aspect=bool(opt.lock_aspect), aspect=float(aspect))
    params = planner.enforce_aspect(initial_params)

    if opt.frames > 0:
        focus_result = render_frame(params, device=DEVICE)
        params = planner.initialize_focus(params, focus_result)

    cmap = get_colormap(opt.colormap)

    if output_config.frame_dir is not None:
        output_config.frame_dir.mkdir(parents=True, exist_ok=True)
    if output_config.gif_path is not None:
        output_config.gif_path.parent.mkdir(parents=True, exist_ok=True)
    if output_config.image_path is not None:
        output_config.image_path.parent.mkdir(parents=True, exist_ok=True)

    def _hex01(hex_color):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            raise ValueError('inside_color must be in the form #RRGGBB.')
        try:
            return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        except ValueError as exc:
            raise ValueError('inside_color must contain only hexadecimal digits.') from exc

    try:
        inside_rgb = _hex01(opt.inside_color)
    except ValueError:
        print(f"Invalid inside_color '{opt.inside_color}', defaulting to black.")
        inside_rgb = (0.0, 0.0, 0.0)

    per_frame_factors = compute_zoom_factors(
        opt.frames,
        opt.zoom_factor,
        final_zoom=opt.final_zoom,
        easing=opt.easing,
    )

    frame_digits = max(3, len(str(max(opt.frames - 1, 0))))
    writers = OutputWriters(
        output_config,
        frame_digits=frame_digits,
        image_format=output_config.image_format,
        show_edges=bool(opt.show_edges),
    )

    final_color_image: PIL.Image.Image | None = None
    final_metadata: SamplingMetadata | None = None
    final_params: RenderParameters | None = None
    coordinates_annotated = False

    try:
        for i in range(opt.frames):
            print("frame {0} out of {1}".format(i, opt.frames), end='\r')
            zoom_factor = per_frame_factors[i] if per_frame_factors.size else opt.zoom_factor
            frame_params = params
            result = render_frame(frame_params, device=DEVICE)
            next_params = (
                planner.update_after_frame(params, result, zoom_factor)
                if i < opt.frames - 1
                else params
            )

            fractal = result.smooth
            iters = result.iterations
            edges = result.edges

            needs_color_array = writers.requires_color_array(i, opt.frames)
            needs_mono = writers.requires_mono_image()

            if not (needs_color_array or needs_mono):
                params = next_params
                continue

            inside = (iters >= opt.max_iterations)
            v = fractal.astype(np.float64, copy=True)
            normalize_mode = getattr(opt, 'normalize', 'outside')
            eps = 1e-12

            selection = v[~inside] if normalize_mode == 'outside' else v

            if selection.size:
                lo = np.percentile(selection, getattr(opt, 'clip_low', 0.5))
                hi = np.percentile(selection, getattr(opt, 'clip_high', 99.5))
                hi = max(hi, lo + eps)
                v = (np.clip(v, lo, hi) - lo) / (hi - lo)
            else:
                v.fill(0.0)

            gamma = getattr(opt, 'gamma', 0.85)
            v = np.clip(v, 0.0, 1.0) ** gamma

            if needs_mono:
                mono = np.uint8(np.clip(v, 0.0, 1.0) * 255)
                mono_img = PIL.Image.fromarray(mono)
                writers.write_mono_output(i, mono_img)

            if not needs_color_array:
                params = next_params
                continue

            cmap_input = 1.0 - v if getattr(opt, 'invert', False) else v
            rgba = np.array(cmap(cmap_input), copy=True)

            for k in (0, 1, 2):
                rgba[..., k] = np.where(inside, inside_rgb[k], rgba[..., k])
            rgba[..., 3] = 1.0
            rgba_uint8 = np.uint8(np.clip(rgba * 255, 0, 255))

            if opt.show_edges:
                edges_rgba = np.uint8(np.stack((edges,) * 4, axis=-1) * 255)
                frame_array = np.concatenate((rgba_uint8, edges_rgba), axis=1)
            else:
                frame_array = rgba_uint8

            need_color_image = writers.requires_color_image(i, opt.frames) or writers.should_store_final_image(i, opt.frames)

            if (
                getattr(opt, "show_coordinates", False)
                and result.metadata is not None
            ):
                annotated_image = annotate_with_coordinates(
                    PIL.Image.fromarray(frame_array),
                    result.metadata,
                    frame_params,
                )
                frame_array = np.array(annotated_image, copy=True)
                color_image = annotated_image if need_color_image else None
                coordinates_annotated = True
            else:
                color_image = PIL.Image.fromarray(frame_array) if need_color_image else None

            writers.write_color_outputs(i, frame_array, color_image)

            if writers.should_store_final_image(i, opt.frames):
                final_color_image = color_image if color_image is not None else PIL.Image.fromarray(frame_array)
                final_metadata = result.metadata
                final_params = frame_params

            params = next_params

    finally:
        writers.close()

    if (
        final_color_image is not None
        and getattr(opt, "show_coordinates", False)
        and final_metadata is not None
        and final_params is not None
        and not coordinates_annotated
    ):
        final_color_image = annotate_with_coordinates(final_color_image, final_metadata, final_params)

    writers.finalize(final_color_image)


if __name__ == '__main__':
    main()

