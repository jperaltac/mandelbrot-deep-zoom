# Import libraries for simulation
import tensorflow as tf
import numpy as np

# Imports for visualization
from PIL import Image
import PIL.Image
from io import BytesIO
from IPython.display import Image, display
from matplotlib import cm  # make sure matplotlib 3.0+ is installed
import matplotlib

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
                        dest='max_iterations', help='maximum number of times to iterate logisitc mapping',
                        metavar='MAX_ITERATIONS', default=2000)

    parser.add_argument('--x-res', type=int,
                        dest='x_res', help='resolution of samples along the x-axis',
                        metavar='X_RES', default=512)

    parser.add_argument('--y-res', type=int,
                        dest='y_res', help='resolution of samples along the y-axis',
                        metavar='SAMPLE_DIR', default=512)

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

    parser.add_argument('--zoom-factor', type=float,
                        dest='zoom_factor', help='the factor to by which to multiply the window size each frame. Choose < 1 for zoom in, >1 for zoom out',
                        metavar='ZOOM_FACTOR', default=0.8)

    parser.add_argument('--frames', type=int,
                        dest='frames', help='number of frames to generate',
                        metavar='FRAMES', default=100)

    parser.add_argument('--save-frames', help='flag to save each frame of the zoom as a iamge',
                        dest='save_frames', action="store_true")

    parser.add_argument('--save-mono', help='flag to save each frame as a monochrome image',
                        dest='save_mono', action="store_true")

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

        # choose an edge point at random to focus on
        _, Z, edges = self.process()
        edge_indices = np.argwhere(edges == 1)
        zoom_center = edge_indices[np.random.choice(len(edge_indices))]

        # recenter to the focus point
        self.x_center = Z[zoom_center[0], zoom_center[1]].real
        self.y_center = Z[zoom_center[0], zoom_center[1]].imag

    @tf.function
    def _step(self, zs, xs, ns):
        """Perform a single Mandelbrot iteration."""
        zs = zs * zs + xs
        az = tf.abs(zs)
        not_diverged = az < HORIZON
        ns = ns + tf.cast(not_diverged, tf.int32)
        return zs, ns, not_diverged

    @tf.function
    def _run(self, xs, zs, ns):
        """Iterate the Mandelbrot formula using a TensorFlow while loop."""
        i = tf.constant(0, dtype=tf.int32)
        not_diverged = tf.ones_like(ns, tf.bool)

        def cond(i, zs, ns, not_diverged):
            return tf.less(i, self.max_n)

        def body(i, zs, ns, not_diverged):
            zs, ns, not_diverged = self._step(zs, xs, ns)
            return i + 1, zs, ns, not_diverged

        return tf.while_loop(cond, body, [i, zs, ns, not_diverged])

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

            iterations, zs, ns, not_diverged = self._run(xs, zs, ns)
            iter_matrix = tf.fill(tf.shape(ns), iterations)
            mask = tf.cast(tf.clip_by_value(iter_matrix - ns, 0, 1), tf.bool)
            edges = tf.math.logical_xor(tf.roll(mask, 1, axis=0), mask)

        return (iter_matrix - ns).numpy(), Z.numpy(), edges.numpy()

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
    images = []
    # print(np.max(cm.twilight_shifted(np.concatenate(image_generator.next_image(zoom_factor=0.8), axis=1))))

    if opt.save_frames or opt.save_mono:
        import os
        try:
            os.mkdir(opt.frames_path)
        except Exception:
            pass

    for i in range(opt.frames):
        print("frame {0} out of {1}".format(i, opt.frames), end='\r')
        fractal, edges = image_generator.next_image(zoom_factor=opt.zoom_factor)

        if opt.save_mono:
            img = PIL.Image.fromarray(
                np.uint8(255 * (np.abs((fractal % 512) - 255) / 256)))
            img.save(os.path.join(opt.frames_path, 'mono{0:03d}.{1}'.format(i, opt.format)))

        fractal = np.uint8(cm.twilight_shifted(fractal % 512) * 255)
        if opt.show_edges:
            edges = np.uint8(np.stack((edges,)*4, axis=-1)*255)
            img = PIL.Image.fromarray(np.concatenate((fractal, edges), axis=1))
            images.append(img)
            if opt.save_frames:
                img.save(os.path.join(opt.frames_path, 'frame{0:03d}.{1}'.format(i, opt.format)))
        else:
            img = PIL.Image.fromarray(fractal)
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

