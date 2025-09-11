# Mandelbrot Zoomer

A Python routine to generate deep zoom animations of the Mandelbrot set using TensorFlow.

<p align="center">
  <img src="examples/movie.gif" height="512px" />
</p>

## Installation

Requires **Python 3.10+**.

### CPU only

```bash
pip install -r requirements.txt
```

### GPU (CUDA)

```bash
pip install -r requirements-gpu.txt
```

TensorFlow automatically uses the first available GPU. To force a CPU run even on a GPU machine:

```bash
CUDA_VISIBLE_DEVICES="" python zoom.py
```

## Usage

Generate a zoom with the default settings:

```bash
python zoom.py
```

Example CPU run with custom resolution and frame count:

```bash
python zoom.py --frames 100 --x-res 512 --y-res 512
```

Example GPU run (assuming a CUDA-capable device is available):

```bash
CUDA_VISIBLE_DEVICES=0 python zoom.py --frames 300 --zoom-factor 0.9
```

The script saves an animated `movie.gif`. Use `--save-frames` to also write individual images to disk.

## Command-line Flags

- `--max-iterations`   Maximum iterations per pixel. Default: `2000`
- `--x-res`            Resolution along the x-axis. Default: `512`
- `--y-res`            Resolution along the y-axis. Default: `512`
- `--x-center`         Starting x coordinate in the complex plane. Default: `-0.75`
- `--y-center`         Starting y coordinate in the complex plane. Default: `0`
- `--x-width`          Initial width of the sample window. Default: `2.5`
- `--y-width`          Initial height of the sample window. Default: `2.5`
- `--zoom-factor`      Factor applied to window size each frame (<1 zooms in). Default: `0.8`
- `--frames`           Number of frames to generate. Default: `100`
- `--save-frames`      Save each colored frame to disk
- `--save-mono`        Save each frame as a monochrome image
- `--format`           File format for saved frames (e.g. `png`, `jpeg`). Default: `png`
- `--frames-path`      Directory to store individual frames. Default: `./frames`
- `--show-edges`       Render edge detection alongside the Mandelbrot zoom

## How does it know where to zoom?

The program detects the edges of the Mandelbrot set and keeps itself centered on them.

<p align="center">
  <img src="examples/edges.gif" height="256px" />
</p>
