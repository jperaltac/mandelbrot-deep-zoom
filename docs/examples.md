# Example playbook

This playbook expands on the usage examples in the main README and focuses on practical presets that you can adapt. Every command can be combined with the flags described in the [command line reference](../README.md#command-line-reference).

## 1. Lightning preview (fast GIF)

Purpose: validate a composition quickly before committing to a longer render.

```bash
python zoom.py \
  --frames 36 \
  --x-res 288 \
  --y-res 288 \
  --zoom-factor 0.85 \
  --max-iterations 1200 \
  --colormap magma
```

- **Render time:** well under 30 seconds on a modern CPU.
- **Result:** lightweight `movie.gif` (≈2 MB) with punchy colours and energetic motion.
- **Tweak it:** increase `--zoom-factor` towards `0.9` for a calmer zoom or drop to `0.8` for punchier acceleration.

## 2. Small fast video for social media

Purpose: create a short MP4 clip that loops smoothly on phones and tablets.

```bash
python zoom.py --frames 75 --zoom-factor 0.9 --x-res 480 --y-res 480 \
              --mode frames --frame-dir frames_social --format png
ffmpeg -framerate 25 -pattern_type glob -i 'frames_social/frame*.png' \
       -c:v libx264 -crf 20 -pix_fmt yuv420p mandelbrot-social.mp4
```

- **Render time:** about 1 minute on CPU, seconds on GPU.
- **Result:** crisp 3 second video (`mandelbrot-social.mp4`) ready for Instagram, Mastodon, or Twitter.
- **Tweak it:** raise `--frames` to 100 for longer clips or use `-crf 18` in ffmpeg to increase quality.

## 3. Ultra-deep dive (desktop wallpaper sequence)

Purpose: capture a slow, intricate dive to extract stills or assemble a long-form animation.

```bash
python zoom.py \
  --frames 600 \
  --zoom-factor 0.975 \
  --x-res 1440 \
  --y-res 1440 \
  --max-iterations 6000 \
  --mode frames \
  --frame-dir deep_frames
```

- **Render time:** can take tens of minutes; recommend using a GPU.
- **Result:** `movie.gif` plus `deep_frames/frameNNN.png` for manual selection.
- **Tweak it:** point `--frame-dir` to a fast SSD; if memory is tight, run in chunks (e.g. 300 frames at a time).

## 4. Static hero image

Purpose: output a single high-resolution still for print or backgrounds.

```bash
python zoom.py \
  --frames 120 \
  --zoom-factor 0.94 \
  --mode image \
  --output renders/final_hero.png \
  --format png \
  --colormap cividis
```

- **Render time:** similar to the default animation; the final frame is written to `renders/final_hero.png`.
- **Result:** high-resolution PNG that captures the final zoom moment.
- **Tweak it:** change `--format` to `jpeg` for lighter files or `--colormap` to `plasma` for warmer hues.

## 5. Edge visualisation split-screen

Purpose: showcase how the auto-centering logic tracks the boundary of the set.

```bash
python zoom.py --frames 180 --zoom-factor 0.92 --show-edges --colormap viridis
```

- **Result:** `movie.gif` with the raw render on the left and Sobel edge detection on the right (see `examples/edges.gif`).
- **Tweak it:** pair with `--show-coordinates` to highlight the origin and window bounds during presentations.

## Useful colour palettes

| Palette | Mood | When to use |
|---------|------|-------------|
| `twilight_shifted` | Balanced blues and oranges | Default palette—great all-rounder. |
| `magma` | Deep reds and oranges | Highlights energy in quick zooms. |
| `inferno` | Vivid yellows and reds | High contrast clips for social media. |
| `cividis` | Colour-blind friendly | Scientific presentations and accessibility. |
| `plasma` | Neon purples and golds | Eye-catching loops and banners. |

Combine palettes with the recipes above to match the tone of your project.
