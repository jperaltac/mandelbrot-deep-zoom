# Mandelbrot Zoomer

Generate deep, smooth Mandelbrot-set zooms from the command line using TensorFlow. The project focuses on producing high quality animations while remaining approachable for quick experiments and short-form clips.

<p align="center">
  <img src="examples/movie.gif" height="512px" alt="Animated Mandelbrot zoom" />
</p>

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Recipes and examples](#recipes-and-examples)
- [Producing tiny fast videos](#producing-tiny-fast-videos)
- [Working with saved frames](#working-with-saved-frames)
- [Command line reference](#command-line-reference)
- [Project roadmap](#project-roadmap)

## Installation

Requires **Python 3.10+**. Pick the requirements file that matches your hardware.

### CPU only

```bash
pip install -r requirements.txt
```

### GPU (CUDA)

```bash
pip install -r requirements-gpu.txt
```

### ROCm (AMD GPUs)

```bash
pip install -r requirements-rocm.txt
```

> **Tip:** TensorFlow automatically uses the first available GPU. To force a CPU run even on a GPU machine, clear the `CUDA_VISIBLE_DEVICES` variable:
>
> ```bash
> CUDA_VISIBLE_DEVICES="" python zoom.py
> ```

## Quick start

1. Install the dependencies.
2. Run the default zoom and open the generated `movie.gif`:
   ```bash
   python zoom.py
   ```
3. Tweak the resolution or duration by adding flags:
   ```bash
   python zoom.py --frames 120 --x-res 768 --y-res 768
   ```

The default animation uses a smooth twilight color palette, stays centered on interesting edges, and writes the animation to `movie.gif` in the working directory.

## Recipes and examples

The following commands demonstrate common workflows. Combine options to tailor the zoom to your taste.

| Goal | Command | Notes |
|------|---------|-------|
| **Fast preview (CPU)** | `python zoom.py --frames 48 --x-res 320 --y-res 320 --zoom-factor 0.9` | Generates a ~2 second GIF that renders quickly on most laptops. |
| **Deep dive (CPU)** | `python zoom.py --frames 480 --zoom-factor 0.97 --max-iterations 4000` | Slow but detailed 20 second animation. Keep an eye on memory usage. |
| **GPU accelerated render** | `CUDA_VISIBLE_DEVICES=0 python zoom.py --frames 360 --x-res 1024 --y-res 1024` | Renders in high resolution if a CUDA-capable GPU is available. |
| **Highlight the fractal edges** | `python zoom.py --show-edges` | Saves a side-by-side animation of the color image and its detected edges (`examples/edges.gif` shows the effect). |
| **Label the coordinates** | `python zoom.py --frames 240 --show-coordinates --zoom-factor 0.94` | Overlays the complex plane window of the final frame. |
| **Try a different palette** | `python zoom.py --colormap plasma` | Any Matplotlib colormap is supported (`viridis`, `magma`, `cividis`, …). |
| **Monochrome sequence for compositing** | `python zoom.py --mode mono --frame-dir ./mono_frames --format png` | Produces grayscale frames ready for further processing. |

### Diving into specific locations

Set the initial window and zoom factor to explore favourite coordinates:

```bash
python zoom.py \
  --x-center -0.743643887037151 \
  --y-center 0.13182590420533 \
  --x-width 0.0006 \
  --y-width 0.0006 \
  --zoom-factor 0.96 \
  --frames 360
```

The script automatically keeps the zoom centered on the most intricate edges that enter the frame.

## Producing tiny fast videos

Short-form clips for social media benefit from a small resolution, fewer frames, and a faster zoom factor. Use the preview preset below as a starting point:

```bash
python zoom.py \
  --frames 60 \
  --x-res 360 \
  --y-res 360 \
  --zoom-factor 0.88 \
  --max-iterations 1500 \
  --colormap inferno
```

This renders a crisp loop in under a minute on most machines and keeps the file size small. For even snappier results, try `--frames 40` with `--zoom-factor 0.82` and a bold palette such as `--colormap plasma`.

To target video-friendly frame rates, export the intermediate frames and assemble them with your favourite encoder:

```bash
python zoom.py --frames 90 --zoom-factor 0.9 --frame-dir frames_fast --mode frames --format png
ffmpeg -framerate 30 -pattern_type glob -i 'frames_fast/frame*.png' \
       -c:v libx264 -pix_fmt yuv420p preview.mp4
```

The resulting `preview.mp4` is a lightweight 3 second clip suitable for social networks, presentations, or messaging apps.

## Working with saved frames

Activating the frame-saving modes turns each rendered frame into a standalone image for post-processing. The CLI currently accepts the historical flags as well as the modern mode-based aliases.

- `python zoom.py --save-frames` writes colour frames to `./frames/frameNNN.png`.
- `python zoom.py --save-mono` writes monochrome counterparts to `./frames/monoNNN.png`.
- `python zoom.py --frames-path custom_dir` changes the destination directory.

You can mix these with `--mode` for future compatibility:

```bash
python zoom.py --mode gif --mode frames --frame-dir ./frames --format jpeg --zoom-factor 0.95
```

Once frames are on disk, tools such as [ffmpeg](https://ffmpeg.org/), [ImageMagick](https://imagemagick.org/) or [moviepy](https://zulko.github.io/moviepy/) can repurpose them into MP4, WebM, or sprite sheets.

## Command line reference

| Flag | Description | Default |
|------|-------------|---------|
| `--max-iterations` | Maximum iterations per pixel. Higher values reveal more detail but cost time. | `2000` |
| `--x-res`, `--y-res` | Output resolution in pixels. | `512`, `512` |
| `--x-center`, `--y-center` | Centre of the zoom in the complex plane. | `-0.75`, `0` |
| `--x-width`, `--y-width` | Initial width and height of the sampled window. Smaller values start closer to the set. | `2.5`, `2.5` |
| `--zoom-factor` | Multiplicative factor applied each frame. Values `< 1` zoom in; values `> 1` zoom out. | `0.8` |
| `--frames` | Number of frames in the animation. | `100` |
| `--colormap` | Matplotlib colour map used for the fractal. | `twilight_shifted` |
| `--format` | Image format for saved frames (`png`, `jpeg`, …). | `png` |
| `--frames-path` | Destination for saved frames when `--save-frames`/`--save-mono` are enabled. | `./frames` |
| `--save-frames` | Save colour frames individually. | Disabled |
| `--save-mono` | Save monochrome frames individually. | Disabled |
| `--show-edges` | Renders Sobel edge detection alongside the zoom. | Disabled |
| `--show-coordinates` | Annotates the final frame with axis bounds and the origin. | Disabled |

## Project roadmap

The CLI is in the process of adopting a more flexible output system. See [`docs/output-modes.md`](docs/output-modes.md) for a deep dive into the proposed redesign, including richer combinations of GIFs, still images, and frame sequences.

Community contributions are welcome—share your favourite zoom coordinates or rendering presets via issues and pull requests!
