# Mandelbrot Zoomer

Generate deep, smooth Mandelbrot-set zooms from the command line using TensorFlow. The project focuses on producing high quality animations while remaining approachable for quick experiments and short-form clips.

Although the CLI exposes descriptive flag names, the number of combinations can feel daunting when you are new to fractal rendering. This guide distils the practical knowledge you need to run your first zoom, explains how the framing maths works, and provides several production-ready presets so you can adapt them without guesswork.

<p align="center">
  <img src="examples/movie.gif" height="512px" alt="Animated Mandelbrot zoom" />
</p>

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
  - [Keeping the image square](#keeping-the-image-square)
- [Recipes and examples](#recipes-and-examples)
  - [Controlling the zoom curve](#controlling-the-zoom-curve)
- [Producing tiny fast videos](#producing-tiny-fast-videos)
- [Colour grading and contrast](#colour-grading-and-contrast)
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
   The script renders 100 frames at 512×512 pixels using a smooth ease-in/ease-out curve and the `twilight_shifted` colour palette.
3. Tweak the resolution or pacing by adding flags:
   ```bash
   python zoom.py --frames 120 --gif-frame-duration 0.06 --x-res 768 --y-res 768
   ```

Every run auto-centres on intricate edges that enter the frame and writes the final animation to `movie.gif` in the working directory unless you override the output path.

### Keeping the image square

The fractal can look stretched if the complex-plane window does not match the pixel aspect ratio. Add `--lock-aspect` whenever you change only one of `--x-res`/`--y-res` or `--x-width`/`--y-width`:

```bash
python zoom.py --x-res 768 --y-res 432 --lock-aspect
```

This keeps `y_width = x_width × (y_res / x_res)` for every frame so that circles remain round.

## Recipes and examples

The following presets cover common workflows and highlight how different options interact. Mix and match them, or use them as starting points for your own renders.

1. **Rapid CPU preview** – confirm the framing and colour palette before a long render.
   ```bash
   python zoom.py --frames 48 --x-res 320 --y-res 320 --zoom-factor 0.9
   ```
   Produces a ~2 second GIF that renders in well under a minute on a laptop while keeping enough detail to judge the composition.

2. **Balanced default-plus** – extend the stock animation for social posts without overwhelming render times.
   ```bash
   python zoom.py --frames 180 --x-res 640 --y-res 640 --zoom-factor 0.92 --max-iterations 2600
   ```
   Slightly increases the resolution and frame count, revealing finer filaments while remaining CPU friendly.

3. **Ultra-deep dive** – explore fine structure for desktop wallpapers or archival frames.
   ```bash
   python zoom.py --frames 480 --zoom-factor 0.97 --max-iterations 4000 --lock-aspect
   ```
   Keeps the aspect ratio square while the high iteration cap resolves thin tendrils. Expect a long render unless a GPU is available.

4. **High-resolution GPU render** – lean on CUDA acceleration for crisp output.
   ```bash
   CUDA_VISIBLE_DEVICES=0 python zoom.py --frames 360 --x-res 1024 --y-res 1024 --zoom-factor 0.94
   ```
   Ensures TensorFlow selects the first GPU and ups the resolution to 1K² for publication-ready animations.

5. **Edge detective** – visualise how the auto-centering keeps interesting geometry in view.
   ```bash
   python zoom.py --frames 180 --zoom-factor 0.92 --show-edges --colormap viridis
   ```
   Generates a split-screen GIF with Sobel edge detection on the right (`examples/edges.gif` shows the layout).

6. **Annotated lecture clip** – add overlays for workshops or tutorials.
   ```bash
   python zoom.py --frames 240 --show-coordinates --zoom-factor 0.94 --x-res 768 --y-res 432 --lock-aspect
   ```
   Presents the bounds of the complex plane on the final frame while `--lock-aspect` keeps circles round even at a 16:9 resolution.

7. **Palette swapper** – quickly preview Matplotlib colour maps.
   ```bash
   python zoom.py --frames 120 --zoom-factor 0.9 --colormap plasma
   ```
   Any palette supported by Matplotlib (`viridis`, `inferno`, `cividis`, …) can be slotted in to shape the mood of the render.

8. **Monochrome compositing pass** – prepare greyscale layers for external pipelines.
   ```bash
   python zoom.py --mode mono --mode frames --frame-dir ./mono_frames --format png --frames 200
   ```
   Writes individual monochrome PNGs to `./mono_frames` and keeps the GIF, making it easy to grade or composite in other tools.

### Controlling the zoom curve

Two parameters define how aggressively the window shrinks:

- `--zoom-factor` multiplies the window size every frame (values `< 1` zoom in, values `> 1` zoom out).
- `--final-zoom` lets you state the total scale of the last frame—useful when you know you want to end at, say, `1e-6` of the original width. When provided, the script computes a matching per-frame factor and ignores `--zoom-factor`.

For non-linear motion, pick an easing curve. The default `ease` acceleration pattern starts gently, speeds up mid-way, and glides to the final scale. Switch to `linear` for constant speed, or `ease-in`/`ease-out` when you want dramatic anticipation or a lingering finish:

```bash
python zoom.py --frames 240 --final-zoom 1e-5 --easing ease
python zoom.py --frames 180 --final-zoom 1e-4 --easing linear
python zoom.py --frames 150 --zoom-factor 0.86 --easing ease-out
```

When `--final-zoom` is present, the script derives a matching per-frame factor and ignores `--zoom-factor`, guaranteeing the final frame lands on the requested scale.

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

## Colour grading and contrast

`zoom.py` provides several post-processing controls to fine-tune the look of the render without editing the frames afterwards:

- `--normalize all` considers every sample (inside and outside the set) when building the histogram. The default `outside`
  ignores points that never escape, which keeps the background dark and emphasises filaments.
- `--gamma` lifts mid-tones after normalisation. Values below `1.0` increase contrast; values above `1.0` flatten it.
- `--clip-low`/`--clip-high` discard the dimmest and brightest percentiles before normalisation. This is helpful when a single
  pixel blows out the scale.
- `--tone-smoothing` blends each frame’s clipping range with its neighbours. Higher values reduce flicker in deep zooms.
- `--invert` flips the colormap, while `--inside-color` (hex) controls the colour used for points inside the Mandelbrot set.

Combine these with Matplotlib colormaps to build a consistent style guide for your animations.

For a punchier render, lift the gamma slightly while trimming bright spikes and switching palettes:

```bash
python zoom.py --frames 150 --zoom-factor 0.9 --gamma 0.95 --clip-low 1 --clip-high 99 --colormap inferno
```

The histogram settings prevent isolated highlights from washing out the colour story, and the palette swap keeps the sequence cohesive.

## Working with saved frames

Activating the frame-saving modes turns each rendered frame into a standalone image for post-processing. The CLI currently accepts the historical flags as well as the modern mode-based aliases.

- `--mode frames` (or the legacy `--save-frames`) writes colour frames to `./frames/frameNNN.png`.
- `--mode mono` (or `--save-mono`) writes monochrome counterparts to `./frames/monoNNN.png`.
- `--frame-dir ./custom_dir` changes the destination directory for any frame-based output. The deprecated `--frames-path` forwards to the same setting.
- `--keep-frames` tells the GIF renderer to keep the intermediate frame sequence alongside the compiled animation.
- `--output` points single-file exports (`--mode gif`/`--mode image`) to a custom location. When you request more than one mode the path becomes a directory that holds every product.

You can mix and match modes to produce several artefacts in a single render:

```bash
python zoom.py --mode gif --mode frames --frame-dir ./frames --format jpeg --zoom-factor 0.95
```

Once frames are on disk, tools such as [ffmpeg](https://ffmpeg.org/), [ImageMagick](https://imagemagick.org/) or [moviepy](https://zulko.github.io/moviepy/) can repurpose them into MP4, WebM, or sprite sheets.

## Command line reference

Para una guía detallada en español con ejemplos ejecutables de cada bandera, revisa [`docs/cli-options.md`](docs/cli-options.md).

### Geometry and motion

| Flag | Description | Default |
|------|-------------|---------|
| `--x-res`, `--y-res` | Output resolution in pixels. | `512`, `512` |
| `--x-width`, `--y-width` | Initial size of the sampled window in the complex plane. Smaller values start closer to the set. | `2.5`, `2.5` |
| `--x-center`, `--y-center` | Centre of the zoom in the complex plane. | `-0.75`, `0` |
| `--lock-aspect` | Keeps `y_width` tied to `x_width × (y_res / x_res)` to avoid stretching. | Disabled |
| `--frames` | Number of frames in the animation. | `100` |
| `--zoom-factor` | Multiplies the window size every frame. `< 1` zooms in; `> 1` zooms out. | `0.8` |
| `--final-zoom` | Total scale applied by the last frame (e.g. `1e-4`). Overrides `--zoom-factor` when present. | Not set |
| `--easing` | Temporal curve for the zoom. Choose `ease` for smooth acceleration or `linear` for a constant rate. | `ease` |

### Rendering and overlays

| Flag | Description | Default |
|------|-------------|---------|
| `--max-iterations` | Maximum iterations per pixel. Higher values reveal more detail but cost time. | `2000` |
| `--colormap` | Matplotlib colour map used for the fractal. | `twilight_shifted` |
| `--invert` | Invert the selected colormap. | Disabled |
| `--inside-color` | Colour for points inside the Mandelbrot set (hex). | `#000000` |
| `--show-edges` | Renders Sobel edge detection alongside the zoom. | Disabled |
| `--show-coordinates` | Annotates the final frame with axis bounds and the origin. | Disabled |

### Tone mapping

| Flag | Description | Default |
|------|-------------|---------|
| `--normalize {outside,all}` | Choose whether only escaping samples or all samples contribute to histogram normalisation. | `outside` |
| `--gamma` | Gamma correction applied after normalisation. | `0.85` |
| `--clip-low`, `--clip-high` | Percentiles that trim the histogram before normalisation. | `0.5`, `99.5` |
| `--tone-smoothing` | EMA strength applied to the clipping range for consecutive frames. `0` disables smoothing. | `0.25` |

### Output management

| Flag | Description | Default |
|------|-------------|---------|
| `--mode` | Output modes to generate. Repeat for multiple outputs: `gif`, `image`, `frames`, `mono`. | `gif` |
| `--output` | Destination for single-file outputs (`gif`, `image`) or the directory that collects every artefact when several modes are requested. | `movie.gif` |
| `--frame-dir` | Directory for saved frames or temporary GIF frames. | `./frames` |
| `--keep-frames` | Preserve the individual frames that were used to build a GIF. | Disabled |
| `--gif-frame-duration` | Seconds that each GIF frame remains on screen. | `0.1` |
| `--format` | File format for image-based outputs (any Pillow-supported extension). | `png` |
| `--save-frames`, `--save-mono` | Legacy equivalents to `--mode frames`/`--mode mono`. | Disabled |
| `--frames-path` | Deprecated alias of `--frame-dir`. | `./frames` |

### Diagnostics

| Flag | Description | Default |
|------|-------------|---------|
| `-v`, `--verbose` | Print TensorFlow and hardware diagnostics during the run. | Disabled |

## Project roadmap

The CLI is in the process of adopting a more flexible output system. See [`docs/output-modes.md`](docs/output-modes.md) for a deep dive into the proposed redesign, including richer combinations of GIFs, still images, and frame sequences. For additional presets and render ideas, browse [`docs/examples.md`](docs/examples.md).

Community contributions are welcome—share your favourite zoom coordinates or rendering presets via issues and pull requests!
