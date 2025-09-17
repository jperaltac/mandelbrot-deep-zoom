# Import libraries for simulation
import tensorflow as tf
import numpy as np

# Imports for visualization
import PIL.Image
from matplotlib import cm  # make sure matplotlib 3.0+ is installed

print("TensorFlow version:", tf.__version__)

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
        print("GPU found, using", gpus[0].name)
    except RuntimeError as e:
        print(e)
        DEVICE = '/CPU:0'
else:
    DEVICE = '/CPU:0'
    print("No GPU found, using CPU")

from argparse import ArgumentParser


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

    parser.add_argument('--save-frames', help='flag to save each frame of the zoom as an image',
                        dest='save_frames', action="store_true")

    parser.add_argument('--save-mono', help='flag to save each frame as a monochrome image',
                        dest='save_mono', action="store_true")

    parser.add_argument('--colormap', type=str,
                        dest='colormap', help='matplotlib colormap to colorize the fractal (e.g. "viridis", "inferno")',
                        metavar='COLORMAP', default='twilight_shifted')

    parser.add_argument('--format', type=str,
                        dest='format', help='file format for the saved frames. Can be any file extension supported by Pillow. Example: \'jpeg\', \'png\', \'bmp\', etc. Default: \'png\'',
                        metavar='FORMAT', default='png')

    parser.add_argument('--frames-path', type=str,
                        dest='frames_path', help='path to the directory in which to store the individual frames',
                        metavar='FRAMES_PATH', default='./frames')

    parser.add_argument('--show-edges', help='render the edge detection beside',
                        dest='show_edges', action="store_true")

    return parser


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
        _, Z, edges = self.process()
        edge_indices = np.argwhere(edges == 1)
        zoom_center = edge_indices[np.random.choice(len(edge_indices))]

        # recenter to the focus point
        self.x_center = Z[zoom_center[0], zoom_center[1]].real
        self.y_center = Z[zoom_center[0], zoom_center[1]].imag

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

        return smooth.numpy(), Z.numpy(), edges.numpy()

    def next_image(self, zoom_factor=0.9):
        fractal, Z, edges = self.process()

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
        self.x_center = Z[zoom_center[0], zoom_center[1]].real
        self.y_center = Z[zoom_center[0], zoom_center[1]].imag

        # zoom
        self.x_width *= np.float64(zoom_factor)
        if getattr(self, 'lock_aspect', False):
            self.y_width = np.float64(self.x_width * self.aspect)
        else:
            self.y_width *= np.float64(zoom_factor)
        return fractal, edges


def main():
    parser = build_parser()
    opt = parser.parse_args()

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
    images = []
    cmap = cm.get_cmap(opt.colormap)

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

    if opt.save_frames or opt.save_mono:
        import os
        try:
            os.mkdir(opt.frames_path)
        except Exception:
            pass

    for i in range(opt.frames):
        print("frame {0} out of {1}".format(i, opt.frames), end='\r')
        zoom_factor = per_frame_factors[i] if per_frame_factors.size else opt.zoom_factor
        fractal, edges = image_generator.next_image(zoom_factor=zoom_factor)

        v = fractal.astype(np.float64, copy=False)
        vmax = np.percentile(v, 99.5) if v.size else 1.0
        v = np.clip(v / (vmax + 1e-9), 0.0, 1.0) ** 0.85

        if opt.save_mono:
            img = PIL.Image.fromarray(np.uint8(v * 255))
            img.save(os.path.join(opt.frames_path, 'mono{0:03d}.{1}'.format(i, opt.format)))

        rgba = np.uint8(cmap(v) * 255)
        if opt.show_edges:
            edges_rgba = np.uint8(np.stack((edges,) * 4, axis=-1) * 255)
            img = PIL.Image.fromarray(np.concatenate((rgba, edges_rgba), axis=1))
            images.append(img)
            if opt.save_frames:
                img.save(os.path.join(opt.frames_path, 'frame{0:03d}.{1}'.format(i, opt.format)))
        else:
            img = PIL.Image.fromarray(rgba)
            images.append(img)
            if opt.save_frames:
                img.save(os.path.join(opt.frames_path, 'frame{0:03d}.{1}'.format(i, opt.format)))

    import imageio
    images[0].save('movie.gif',
                   save_all=True,
                   append_images=images[1:],
                   duration=100,
                   loop=0)


if __name__ == '__main__':
    main()

