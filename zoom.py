# Import libraries for simulation
import tensorflow as tf
import numpy as np

# Imports for visualization
from PIL import Image
import PIL.Image
from io import BytesIO
from IPython.display import Image, display
from matplotlib import cm # make sure matplotlib 3.0+ is installed
import matplotlib

print("TensorFlow 2.0 is required for this notebook")
print("TensorFlow version:", tf.__version__)
from argparse import ArgumentParser

def build_parser():
    parser = ArgumentParser()
    
    parser.add_argument('--max-iterations', type=int,
                        dest='max_iterations', help='maximum number of times to iterate logisitc mapping',
                        metavar='MAX_ITERATIONS', default=2000)
    
    parser.add_argument('--x-res', type=int,
                        dest='x-res', help='resolution of samples along the x-axis',
                        metavar='X_RES', default=512)

    parser.add_argument('--y-res', type=int,
                        dest='y-res', help='resolution of samples along the y-axis',
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

    parser.add_argument('--save-frames', help='flag to save each frame of the zoom as a .png',
                        dest='save_frames', action="store_true")
    
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
        
        self.x_res = np.float128(x_res)
        self.y_res = np.float128(y_res)
        self.x_center = np.float128(x_center)
        self.x_width = np.float128(x_width)
        self.y_center = np.float128(y_center)
        self.y_width = np.float128(y_width)
        self.max_n = max_n
        
        _, Z, edges = self.__process()
        
        # choose an edge point at random to focus on
        edge_indices = np.argwhere(edges == 1)
        zoom_center = edge_indices[np.random.choice(len(edge_indices))]
        
        # recenter to the focus point
        self.x_center = Z[zoom_center[0], zoom_center[1]].real
        self.y_center = Z[zoom_center[0], zoom_center[1]].imag

    def step(self, _xs, _zs, _ns, prev_not_diverged):
        _zs = _zs*_zs + _xs

        az =  tf.abs(_zs)
        not_diverged = az < HORIZON
        _ns = tf.add(_ns, tf.cast(not_diverged, tf.int32))
        # add the antialiasing bias to newly diverged samples
        #newly_diverged = np.logical_xor(not_diverged, prev_not_diverged)
        #antialiasing_bias = -np.log(np.log(az))/np.log(2) + LOG_HORIZON
        #_ns = tf.add(_ns, np.multiply(newly_diverged, antialiasing_bias))
        return _xs, _zs, _ns, np.array(not_diverged)

    def __process(self):
        x_min = self.x_center - self.x_width/2
        x_max = self.x_center + self.x_width/2
        y_min = self.y_center - self.y_width/2
        y_max = self.y_center + self.y_width/2
        h_x = (x_max - x_min) / self.x_res
        h_y = (y_max - y_min) / self.y_res

        X, Y = np.meshgrid(np.r_[x_min:x_max:h_x], np.r_[y_min:y_max:h_y])
        Z = X + 1j*Y

        xs = tf.constant(Z.astype(np.complex128))
        zs = tf.Variable(xs)
        ns = tf.Variable(tf.zeros_like(xs, tf.int32))
        not_diverged = np.bool(np.ones_like(ns))
        
        iterations = 0
#         temp = 2.0
#         thresh = 1e-2

        for i in range(self.max_n): 
            xs, zs, ns, not_diverged = self.step(xs, zs, ns, not_diverged)

#             # double the iterations each time the threshold is not met
#             if ((i & (i-1) == 0) and i != 0):
#                 print(i)
#                 convergence_ratio = np.sum(not_diverged) / not_diverged.size
#                 if (temp - convergence_ratio < thresh):
#                     iterations = i + 1
#                     break
#                 temp = convergence_ratio

            iterations = i + 1
        
        # edge detection
        edges = np.logical_xor(np.roll(np.clip(iterations - np.array(ns), 0 ,1), 1, axis=0), np.clip(iterations - np.array(ns), 0 ,1))
        
        return iterations - np.array(ns), Z, edges
          
    def next_image(self, zoom_factor=0.9):
        fractal, Z, edges = self.__process()
    
        # choose an edge point nearest to the center
        zoom_center = [0, 0]
        for i in range(min(edges.shape[0], edges.shape[1]) // 2):
            mask = np.zeros_like(edges)
            mask[mask.shape[0]//2-i:mask.shape[0]//2+i, mask.shape[1]//2-i:mask.shape[1]//2+i] = 1
            edge_indices = np.argwhere(np.multiply(edges, mask) == 1)
            if edge_indices.size != 0:
                zoom_center = edge_indices[np.random.choice(len(edge_indices))]
                break

        #recenter
        self.x_center = Z[zoom_center[0], zoom_center[1]].real
        self.y_center = Z[zoom_center[0], zoom_center[1]].imag

        #zoom
        self.x_width *= np.float128(zoom_factor)
        self.y_width *= np.float128(zoom_factor)
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
    for i in range(opt.frames):
        print("frame {0} out of {1}".format(i, opt.frames), end='\r')
        fractal, edges = image_generator.next_image(zoom_factor=opt.zoom_factor)

        fractal = np.uint8(cm.twilight_shifted(fractal%512)*255)
        if opt.show_edges:
            edges = np.uint8(np.stack((edges,)*4, axis=-1)*255)
            images.append(PIL.Image.fromarray(np.concatenate((fractal, edges), axis=1)))
        else:
            images.append(PIL.Image.fromarray(fractal))

    if opt.save_frames:
        import os

        # define the name of the directory to be created
        path = "./frames/"
        try:
            os.mkdir(path)
        except:
            pass
            
        for i, image in enumerate(images):
            image.save('./frames/frame{0}.png'.format(i))

    import imageio
    images[0].save('movie.gif',
                   save_all=True,
                   append_images=images[1:],
                   duration=100,
                   loop=0)
    
if __name__ == '__main__':
    main()

