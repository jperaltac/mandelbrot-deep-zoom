from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

EXAMPLES_ROOT = Path("examples/cli-options")
BASE_ARGS = ["--frames", "1", "--mode", "image", "--x-res", "160", "--y-res", "160"]


@dataclass
class Expected:
    path: Path
    is_dir: bool = False


@dataclass
class Example:
    name: str
    args: list[str]
    expected: list[Expected]
    clean: list[Path] | None = None

    def full_args(self) -> list[str]:
        return ["python", "zoom.py", *self.args]


EXAMPLES: list[Example] = [
    Example(
        name="max-iterations",
        args=[*BASE_ARGS, "--max-iterations", "3500", "--output", str(EXAMPLES_ROOT / "max-iterations" / "high-iterations.png")],
        expected=[Expected(EXAMPLES_ROOT / "max-iterations" / "high-iterations.png")],
        clean=[EXAMPLES_ROOT / "max-iterations"],
    ),
    Example(
        name="x-res",
        args=[*BASE_ARGS, "--x-res", "240", "--output", str(EXAMPLES_ROOT / "x-res" / "wide-resolution.png")],
        expected=[Expected(EXAMPLES_ROOT / "x-res" / "wide-resolution.png")],
        clean=[EXAMPLES_ROOT / "x-res"],
    ),
    Example(
        name="y-res",
        args=[*BASE_ARGS, "--y-res", "96", "--output", str(EXAMPLES_ROOT / "y-res" / "short-resolution.png")],
        expected=[Expected(EXAMPLES_ROOT / "y-res" / "short-resolution.png")],
        clean=[EXAMPLES_ROOT / "y-res"],
    ),
    Example(
        name="x-center",
        args=[*BASE_ARGS, "--x-center", "-1.401155", "--output", str(EXAMPLES_ROOT / "x-center" / "period-three.png")],
        expected=[Expected(EXAMPLES_ROOT / "x-center" / "period-three.png")],
        clean=[EXAMPLES_ROOT / "x-center"],
    ),
    Example(
        name="x-width",
        args=[*BASE_ARGS, "--x-width", "0.6", "--output", str(EXAMPLES_ROOT / "x-width" / "narrow-window.png")],
        expected=[Expected(EXAMPLES_ROOT / "x-width" / "narrow-window.png")],
        clean=[EXAMPLES_ROOT / "x-width"],
    ),
    Example(
        name="y-center",
        args=[*BASE_ARGS, "--y-center", "0.35", "--output", str(EXAMPLES_ROOT / "y-center" / "upper-plane.png")],
        expected=[Expected(EXAMPLES_ROOT / "y-center" / "upper-plane.png")],
        clean=[EXAMPLES_ROOT / "y-center"],
    ),
    Example(
        name="y-width",
        args=[*BASE_ARGS, "--y-width", "1.2", "--output", str(EXAMPLES_ROOT / "y-width" / "tall-window.png")],
        expected=[Expected(EXAMPLES_ROOT / "y-width" / "tall-window.png")],
        clean=[EXAMPLES_ROOT / "y-width"],
    ),
    Example(
        name="lock-aspect",
        args=[*BASE_ARGS, "--x-res", "256", "--lock-aspect", "--output", str(EXAMPLES_ROOT / "lock-aspect" / "locked.png")],
        expected=[Expected(EXAMPLES_ROOT / "lock-aspect" / "locked.png")],
        clean=[EXAMPLES_ROOT / "lock-aspect"],
    ),
    Example(
        name="zoom-factor",
        args=[
            "--frames",
            "6",
            "--mode",
            "image",
            "--x-res",
            "160",
            "--y-res",
            "160",
            "--zoom-factor",
            "0.7",
            "--output",
            str(EXAMPLES_ROOT / "zoom-factor" / "fast-zoom.png"),
        ],
        expected=[Expected(EXAMPLES_ROOT / "zoom-factor" / "fast-zoom.png")],
        clean=[EXAMPLES_ROOT / "zoom-factor"],
    ),
    Example(
        name="final-zoom",
        args=[
            "--frames",
            "5",
            "--mode",
            "image",
            "--x-res",
            "160",
            "--y-res",
            "160",
            "--final-zoom",
            "1e-3",
            "--output",
            str(EXAMPLES_ROOT / "final-zoom" / "target-scale.png"),
        ],
        expected=[Expected(EXAMPLES_ROOT / "final-zoom" / "target-scale.png")],
        clean=[EXAMPLES_ROOT / "final-zoom"],
    ),
    Example(
        name="easing",
        args=[
            "--frames",
            "6",
            "--mode",
            "image",
            "--x-res",
            "160",
            "--y-res",
            "160",
            "--zoom-factor",
            "0.7",
            "--easing",
            "linear",
            "--output",
            str(EXAMPLES_ROOT / "easing" / "linear-ease.png"),
        ],
        expected=[Expected(EXAMPLES_ROOT / "easing" / "linear-ease.png")],
        clean=[EXAMPLES_ROOT / "easing"],
    ),
    Example(
        name="frames",
        args=[
            "--frames",
            "12",
            "--mode",
            "gif",
            "--x-res",
            "160",
            "--y-res",
            "160",
            "--zoom-factor",
            "0.85",
            "--output",
            str(EXAMPLES_ROOT / "frames" / "twelve-frames.gif"),
        ],
        expected=[Expected(EXAMPLES_ROOT / "frames" / "twelve-frames.gif")],
        clean=[EXAMPLES_ROOT / "frames"],
    ),
    Example(
        name="mode",
        args=[
            "--frames",
            "1",
            "--mode",
            "image",
            "--mode",
            "mono",
            "--frame-dir",
            str(EXAMPLES_ROOT / "mode" / "mono"),
            "--x-res",
            "160",
            "--y-res",
            "160",
            "--output",
            str(EXAMPLES_ROOT / "mode" / "single-frame.png"),
        ],
        expected=[
            Expected(EXAMPLES_ROOT / "mode" / "single-frame.png"),
            Expected(EXAMPLES_ROOT / "mode" / "mono", is_dir=True),
        ],
        clean=[EXAMPLES_ROOT / "mode"],
    ),
    Example(
        name="output",
        args=[*BASE_ARGS, "--output", str(EXAMPLES_ROOT / "output" / "custom-name.png")],
        expected=[Expected(EXAMPLES_ROOT / "output" / "custom-name.png")],
        clean=[EXAMPLES_ROOT / "output"],
    ),
    Example(
        name="frame-dir",
        args=[
            "--frames",
            "1",
            "--mode",
            "frames",
            "--frame-dir",
            str(EXAMPLES_ROOT / "frame-dir" / "frames"),
            "--x-res",
            "160",
            "--y-res",
            "160",
        ],
        expected=[Expected(EXAMPLES_ROOT / "frame-dir" / "frames", is_dir=True)],
        clean=[EXAMPLES_ROOT / "frame-dir"],
    ),
    Example(
        name="keep-frames",
        args=[
            "--frames",
            "4",
            "--mode",
            "gif",
            "--keep-frames",
            "--frame-dir",
            str(EXAMPLES_ROOT / "keep-frames" / "frames"),
            "--x-res",
            "160",
            "--y-res",
            "160",
            "--output",
            str(EXAMPLES_ROOT / "keep-frames" / "zoom.gif"),
        ],
        expected=[
            Expected(EXAMPLES_ROOT / "keep-frames" / "zoom.gif"),
            Expected(EXAMPLES_ROOT / "keep-frames" / "frames", is_dir=True),
        ],
        clean=[EXAMPLES_ROOT / "keep-frames"],
    ),
    Example(
        name="gif-frame-duration",
        args=[
            "--frames",
            "6",
            "--mode",
            "gif",
            "--gif-frame-duration",
            "0.2",
            "--x-res",
            "160",
            "--y-res",
            "160",
            "--output",
            str(EXAMPLES_ROOT / "gif-frame-duration" / "slow.gif"),
        ],
        expected=[Expected(EXAMPLES_ROOT / "gif-frame-duration" / "slow.gif")],
        clean=[EXAMPLES_ROOT / "gif-frame-duration"],
    ),
    Example(
        name="save-frames",
        args=[
            "--frames",
            "3",
            "--save-frames",
            "--frame-dir",
            str(EXAMPLES_ROOT / "save-frames" / "frames"),
            "--x-res",
            "160",
            "--y-res",
            "160",
        ],
        expected=[
            Expected(EXAMPLES_ROOT / "save-frames" / "frames", is_dir=True),
        ],
        clean=[EXAMPLES_ROOT / "save-frames"],
    ),
    Example(
        name="save-mono",
        args=[
            "--frames",
            "3",
            "--save-mono",
            "--frame-dir",
            str(EXAMPLES_ROOT / "save-mono" / "mono"),
            "--x-res",
            "160",
            "--y-res",
            "160",
        ],
        expected=[
            Expected(EXAMPLES_ROOT / "save-mono" / "mono", is_dir=True),
        ],
        clean=[EXAMPLES_ROOT / "save-mono"],
    ),
    Example(
        name="colormap",
        args=[*BASE_ARGS, "--colormap", "inferno", "--output", str(EXAMPLES_ROOT / "colormap" / "inferno.png")],
        expected=[Expected(EXAMPLES_ROOT / "colormap" / "inferno.png")],
        clean=[EXAMPLES_ROOT / "colormap"],
    ),
    Example(
        name="format",
        args=[
            "--frames",
            "1",
            "--mode",
            "image",
            "--x-res",
            "160",
            "--y-res",
            "160",
            "--format",
            "webp",
            "--output",
            str(EXAMPLES_ROOT / "format" / "custom.webp"),
        ],
        expected=[Expected(EXAMPLES_ROOT / "format" / "custom.webp")],
        clean=[EXAMPLES_ROOT / "format"],
    ),
    Example(
        name="frames-path",
        args=[
            "--frames",
            "1",
            "--mode",
            "frames",
            "--frames-path",
            str(EXAMPLES_ROOT / "frames-path" / "frames"),
            "--x-res",
            "160",
            "--y-res",
            "160",
        ],
        expected=[Expected(EXAMPLES_ROOT / "frames-path" / "frames", is_dir=True)],
        clean=[EXAMPLES_ROOT / "frames-path"],
    ),
    Example(
        name="show-edges",
        args=[*BASE_ARGS, "--show-edges", "--output", str(EXAMPLES_ROOT / "show-edges" / "edges.png")],
        expected=[Expected(EXAMPLES_ROOT / "show-edges" / "edges.png")],
        clean=[EXAMPLES_ROOT / "show-edges"],
    ),
    Example(
        name="show-coordinates",
        args=[*BASE_ARGS, "--show-coordinates", "--output", str(EXAMPLES_ROOT / "show-coordinates" / "annotated.png")],
        expected=[Expected(EXAMPLES_ROOT / "show-coordinates" / "annotated.png")],
        clean=[EXAMPLES_ROOT / "show-coordinates"],
    ),
    Example(
        name="normalize",
        args=[*BASE_ARGS, "--normalize", "all", "--output", str(EXAMPLES_ROOT / "normalize" / "all-samples.png")],
        expected=[Expected(EXAMPLES_ROOT / "normalize" / "all-samples.png")],
        clean=[EXAMPLES_ROOT / "normalize"],
    ),
    Example(
        name="gamma",
        args=[*BASE_ARGS, "--gamma", "1.1", "--output", str(EXAMPLES_ROOT / "gamma" / "bright.png")],
        expected=[Expected(EXAMPLES_ROOT / "gamma" / "bright.png")],
        clean=[EXAMPLES_ROOT / "gamma"],
    ),
    Example(
        name="clip-low",
        args=[*BASE_ARGS, "--clip-low", "5", "--output", str(EXAMPLES_ROOT / "clip-low" / "higher-floor.png")],
        expected=[Expected(EXAMPLES_ROOT / "clip-low" / "higher-floor.png")],
        clean=[EXAMPLES_ROOT / "clip-low"],
    ),
    Example(
        name="clip-high",
        args=[*BASE_ARGS, "--clip-high", "90", "--output", str(EXAMPLES_ROOT / "clip-high" / "lower-ceiling.png")],
        expected=[Expected(EXAMPLES_ROOT / "clip-high" / "lower-ceiling.png")],
        clean=[EXAMPLES_ROOT / "clip-high"],
    ),
    Example(
        name="tone-smoothing",
        args=[
            "--frames",
            "5",
            "--mode",
            "image",
            "--x-res",
            "160",
            "--y-res",
            "160",
            "--tone-smoothing",
            "0.6",
            "--output",
            str(EXAMPLES_ROOT / "tone-smoothing" / "smoothed.png"),
        ],
        expected=[Expected(EXAMPLES_ROOT / "tone-smoothing" / "smoothed.png")],
        clean=[EXAMPLES_ROOT / "tone-smoothing"],
    ),
    Example(
        name="invert",
        args=[*BASE_ARGS, "--invert", "--output", str(EXAMPLES_ROOT / "invert" / "inverted.png")],
        expected=[Expected(EXAMPLES_ROOT / "invert" / "inverted.png")],
        clean=[EXAMPLES_ROOT / "invert"],
    ),
    Example(
        name="inside-color",
        args=[*BASE_ARGS, "--inside-color", "#0a3ba0", "--output", str(EXAMPLES_ROOT / "inside-color" / "custom-interior.png")],
        expected=[Expected(EXAMPLES_ROOT / "inside-color" / "custom-interior.png")],
        clean=[EXAMPLES_ROOT / "inside-color"],
    ),
    Example(
        name="verbose",
        args=[*BASE_ARGS, "--verbose", "--output", str(EXAMPLES_ROOT / "verbose" / "diagnostic.png")],
        expected=[Expected(EXAMPLES_ROOT / "verbose" / "diagnostic.png")],
        clean=[EXAMPLES_ROOT / "verbose"],
    ),
]


def _ensure_clean(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def _prepare(example: Example) -> None:
    clean = example.clean or []
    _ensure_clean(clean)
    for expected in example.expected:
        expected.path.parent.mkdir(parents=True, exist_ok=True)


def _verify(example: Example) -> None:
    for expected in example.expected:
        if expected.is_dir:
            if not expected.path.is_dir():
                raise RuntimeError(f"Expected directory {expected.path} was not created")
            if not any(expected.path.iterdir()):
                raise RuntimeError(f"Directory {expected.path} is empty")
        else:
            if not expected.path.is_file():
                raise RuntimeError(f"Expected file {expected.path} was not created")


def main() -> None:
    EXAMPLES_ROOT.mkdir(parents=True, exist_ok=True)
    for example in EXAMPLES:
        print(f"\n[cli-example] {example.name}")
        _prepare(example)
        completed = subprocess.run(example.full_args(), check=True)
        if completed.returncode != 0:
            raise RuntimeError(f"Example {example.name} failed with {completed.returncode}")
        _verify(example)
    print("\nAll CLI examples generated successfully.")


if __name__ == "__main__":
    main()
