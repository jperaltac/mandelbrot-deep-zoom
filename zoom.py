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
import imageio

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


HORIZON = 4
#LOG_HORIZON = np.log(np.log(HORIZON))/np.log(2)


class mandelbrot_generator:
    def __init__(self,
                x_res=512,
                y_res=512,
                x_center=-0.75,
                x_width=2.5,
                y_center=0,
                y_width=2.5,
                max_n=2000):

        # Use float64 so that the values can be transferred to the GPU. The
        # previous implementation relied on float128 which is not supported by
        # most accelerators.
        self.x_res = np.float64(x_res)
        self.y_res = np.float64(y_res)
        self.x_center = np.float64(x_center)
        self.x_width = np.float64(x_width)
        self.y_center = np.float64(y_center)
        self.y_width = np.float64(y_width)
        self.max_n = int(max_n)

        # Aspect locking support
        self.lock_aspect = getattr(self, 'lock_aspect', False)
        self.aspect = np.float64(self.y_res / self.x_res) if self.x_res != 0 else np.float64(1.0)
        if self.lock_aspect:
            self.y_width = np.float64(self.x_width * self.aspect)

        # choose an edge point at random to focus on
        _, _, edges = self.process()
        edge_indices = np.argwhere(edges == 1)
        zoom_center = edge_indices[np.random.choice(len(edge_indices))]

        # recenter to the focus point
        x_c, y_c = self._pixel_to_complex(zoom_center[0], zoom_center[1])
        self.x_center = np.float64(x_c)
        self.y_center = np.float64(y_c)

    def _pixel_to_complex(self, row, col):
        """Convert pixel indices to complex plane coordinates for the last render."""
        if not hasattr(self, '_x_min') or not hasattr(self, '_y_min'):
            raise RuntimeError('Sampling bounds have not been initialized.')
        x_step = getattr(self, '_x_step', None)
        y_step = getattr(self, '_y_step', None)
        x_res = getattr(self, '_x_res_int', None)
        y_res = getattr(self, '_y_res_int', None)
        if x_step is None or y_step is None or x_res is None or y_res is None:
            raise RuntimeError('Sampling metadata is unavailable.')

        row = int(row)
        col = int(col)
        if x_res > 1:
            x = self._x_min + np.float64(col) * x_step
        else:
            x = self._x_min
        if y_res > 1:
            y = self._y_min + np.float64(row) * y_step
        else:
            y = self._y_min
        return np.float64(x), np.float64(y)

    @tf.function
    def _step(self, zs, xs, ns, active):
        """Perform a single Mandelbrot iteration for points that have not yet diverged."""
        zs_new = zs * zs + xs
        zs = tf.where(active, zs_new, zs)
        ns = ns + tf.cast(active, tf.int32)
        az = tf.abs(zs)
        new_active = tf.logical_and(active, az < HORIZON)
        return zs, ns, new_active

    @tf.function
    def _run(self, xs, zs, ns):
        """Iterate the Mandelbrot formula using a TensorFlow while loop."""
        i = tf.constant(0, dtype=tf.int32)
        active = tf.ones_like(ns, tf.bool)

        def cond(i, zs, ns, active):
            return tf.logical_and(tf.less(i, self.max_n), tf.reduce_any(active))

        def body(i, zs, ns, active):
            zs, ns, active = self._step(zs, xs, ns, active)
            return i + 1, zs, ns, active

        return tf.while_loop(cond, body, [i, zs, ns, active])

    def process(self):
        x_min = self.x_center - self.x_width / 2
        x_max = self.x_center + self.x_width / 2
        y_min = self.y_center - self.y_width / 2
        y_max = self.y_center + self.y_width / 2

        x_res_int = max(int(self.x_res), 1)
        y_res_int = max(int(self.y_res), 1)
        self._x_min = np.float64(x_min)
        self._y_min = np.float64(y_min)
        self._x_res_int = x_res_int
        self._y_res_int = y_res_int
        self._x_step = np.float64((x_max - x_min) / (x_res_int - 1)) if x_res_int > 1 else np.float64(0.0)
        self._y_step = np.float64((y_max - y_min) / (y_res_int - 1)) if y_res_int > 1 else np.float64(0.0)

        with tf.device(DEVICE):
            x = tf.linspace(x_min, x_max, int(self.x_res))
            y = tf.linspace(y_min, y_max, int(self.y_res))
            X, Y = tf.meshgrid(x, y)
            Z = tf.complex(X, Y)
            xs = tf.identity(Z)
            zs = tf.identity(xs)
            ns = tf.zeros_like(xs, tf.int32)

            _, zs, ns, _ = self._run(xs, zs, ns)

            az = tf.abs(zs)
            eps = tf.constant(1e-12, dtype=az.dtype)
            az_safe = tf.maximum(az, tf.constant(1.0, dtype=az.dtype) + eps)
            log_az = tf.math.log(az_safe)
            log_log_az = tf.math.log(tf.maximum(log_az, eps))
            ns_float = tf.cast(ns, tf.float64)
            log2 = tf.math.log(tf.constant(2.0, dtype=az.dtype))
            smooth_escape = ns_float + tf.constant(1.0, dtype=ns_float.dtype) - log_log_az / log2
            escaped = tf.less(ns, self.max_n)
            smooth = tf.where(escaped, smooth_escape, tf.cast(self.max_n, tf.float64))

            mask = tf.cast(
                tf.clip_by_value(
                    tf.fill(tf.shape(ns), tf.cast(self.max_n, ns.dtype)) - ns,
                    0,
                    1,
                ),
                tf.bool,
            )
            edges = tf.math.logical_xor(tf.roll(mask, 1, axis=0), mask)

        return smooth.numpy(), ns.numpy(), edges.numpy()

    def next_image(self, zoom_factor=0.9):
        fractal, ns, edges = self.process()

        # choose an edge point nearest to the center
        zoom_center = [0, 0]
        for i in range(min(edges.shape[0], edges.shape[1]) // 2):
            mask = np.zeros_like(edges)
            mask[mask.shape[0]//2-i:mask.shape[0]//2+i,
                 mask.shape[1]//2-i:mask.shape[1]//2+i] = 1
            edge_indices = np.argwhere(np.multiply(edges, mask) == 1)
            if edge_indices.size != 0:
                zoom_center = edge_indices[np.random.choice(len(edge_indices))]
                break

        # recenter
        x_c, y_c = self._pixel_to_complex(zoom_center[0], zoom_center[1])
        self.x_center = np.float64(x_c)
        self.y_center = np.float64(y_c)

        # zoom
        self.x_width *= np.float64(zoom_factor)
        if getattr(self, 'lock_aspect', False):
            self.y_width = np.float64(self.x_width * self.aspect)
        else:
            self.y_width *= np.float64(zoom_factor)
        return fractal, ns, edges


def main():
    parser = build_parser()
    opt = parser.parse_args()

    output_config = resolve_output_config(opt, parser)

    global VERBOSE
    VERBOSE = bool(opt.verbose)

    image_generator = mandelbrot_generator(opt.x_res,
                                           opt.y_res,
                                           opt.x_center,
                                           opt.x_width,
                                           opt.y_center,
                                           opt.y_width,
                                           opt.max_iterations)
    image_generator.lock_aspect = opt.lock_aspect
    image_generator.aspect = np.float64(image_generator.y_res / image_generator.x_res) if image_generator.x_res != 0 else np.float64(1.0)
    if opt.lock_aspect:
        image_generator.y_width = np.float64(image_generator.x_width * image_generator.aspect)
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

    def ease_in_out(t):
        return 3 * t ** 2 - 2 * t ** 3

    if opt.final_zoom is not None and opt.final_zoom > 0:
        total_frames = max(1, opt.frames)
        logZ = np.log(opt.final_zoom)
        easing_mode = opt.easing.lower()
        ease = (lambda u: u) if easing_mode == 'linear' else ease_in_out
        if total_frames == 1:
            alphas = np.array([1.0], dtype=np.float64)
        else:
            alphas = np.array([ease(i / (total_frames - 1)) for i in range(total_frames)], dtype=np.float64)
        alphas = np.clip(alphas, 0.0, 1.0)
        inc = np.diff(np.concatenate(([0.0], alphas)))
        per_frame_logs = inc * logZ
        if opt.frames < total_frames:
            per_frame_logs = per_frame_logs[:opt.frames]
    else:
        per_frame_logs = np.full(opt.frames, np.log(opt.zoom_factor), dtype=np.float64) if opt.frames > 0 else np.array([], dtype=np.float64)

    per_frame_factors = np.exp(per_frame_logs) if per_frame_logs.size else np.array([], dtype=np.float64)

    frame_digits = max(3, len(str(max(opt.frames - 1, 0))))
    writers = OutputWriters(
        output_config,
        frame_digits=frame_digits,
        image_format=output_config.image_format,
        show_edges=bool(opt.show_edges),
    )

    final_color_image: PIL.Image.Image | None = None

    try:
        for i in range(opt.frames):
            print("frame {0} out of {1}".format(i, opt.frames), end='\r')
            zoom_factor = per_frame_factors[i] if per_frame_factors.size else opt.zoom_factor
            fractal, iters, edges = image_generator.next_image(zoom_factor=zoom_factor)

            needs_color_array = writers.requires_color_array(i, opt.frames)
            needs_mono = writers.requires_mono_image()

            if not (needs_color_array or needs_mono):
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
            color_image = PIL.Image.fromarray(frame_array) if need_color_image else None

            writers.write_color_outputs(i, frame_array, color_image)

            if writers.should_store_final_image(i, opt.frames):
                final_color_image = color_image if color_image is not None else PIL.Image.fromarray(frame_array)

    finally:
        writers.close()

    writers.finalize(final_color_image)


if __name__ == '__main__':
    main()

