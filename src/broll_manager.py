"""
B-Roll Manager — generates abstract tech visuals for TikTok video segments.

Supports two backends:
  1. API-based: calls an image/video generation API (set BROLL_API_KEY env var)
  2. FFmpeg fallback: generates procedural abstract visuals locally

The fallback produces visually interesting clips using ffmpeg filters:
mandelbrot zooms, cellular automata, plasma gradients, and noise patterns.
"""

import os
import random
import subprocess
from os import path


# ── FFmpeg procedural generators ─────────────────────────────────────────────

def _run_ffmpeg(cmd: list[str], label: str = ""):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[B-Roll] ffmpeg error ({label}): {result.stderr[-500:]}")
        raise RuntimeError(f"B-roll generation failed: {label}")


def _gen_mandelbrot_zoom(output_path: str, duration: float, width: int, height: int):
    """Mandelbrot fractal zoom — looks like an abstract digital dive."""
    _run_ffmpeg([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", (
            f"mandelbrot=s={width}x{height}:maxiter=200:rate=30"
            f":start_scale=2:end_scale=0.001"
        ),
        "-t", str(duration),
        "-vf", "colorbalance=rs=0.3:gs=-0.1:bs=0.4,eq=contrast=1.3:brightness=-0.05",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
        output_path,
    ], "mandelbrot")


def _gen_plasma_gradient(output_path: str, duration: float, width: int, height: int):
    """Animated plasma/gradient pattern with neon color grading."""
    _run_ffmpeg([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", (
            f"cellauto=s={width}x{height}:rule=110:rate=30:ratio=0.5"
        ),
        "-t", str(duration),
        "-vf", (
            f"scale={width}:{height}:flags=neighbor,"
            "colorbalance=rs=0.5:gs=-0.3:bs=0.6,"
            "eq=contrast=1.5:brightness=-0.1:saturation=2.0,"
            "gblur=sigma=2"
        ),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
        output_path,
    ], "plasma")


def _gen_noise_glitch(output_path: str, duration: float, width: int, height: int):
    """Digital noise/static with color shift — glitch aesthetic."""
    _run_ffmpeg([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=#0A0A0A:s={width}x{height}:d={duration}:rate=30",
        "-f", "lavfi",
        "-i", f"nullsrc=s={width}x{height}:d={duration}:rate=30,geq=random(1)*255:128:128",
        "-filter_complex", (
            "[1:v]colorbalance=rs=0.4:gs=-0.2:bs=0.5[noise];"
            "[0:v][noise]blend=all_mode=screen:all_opacity=0.3,"
            "eq=contrast=1.8:brightness=-0.15"
        ),
        "-t", str(duration),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
        output_path,
    ], "noise_glitch")


def _gen_waveform(output_path: str, duration: float, width: int, height: int):
    """Animated sine waveform — abstract data visualization look."""
    _run_ffmpeg([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"sine=frequency=2:sample_rate=44100:duration={duration}",
        "-filter_complex", (
            f"[0:a]showwaves=s={width}x{height}:mode=cline:rate=30:colors=0x39FF14|0x00D4FF|0xFF073A,"
            "colorbalance=rs=0.1:bs=0.3,"
            "eq=brightness=-0.2:contrast=1.3[v]"
        ),
        "-map", "[v]",
        "-t", str(duration),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
        "-an",
        output_path,
    ], "waveform")


# Pool of generators for variety
_GENERATORS = [
    _gen_mandelbrot_zoom,
    _gen_plasma_gradient,
    _gen_noise_glitch,
    _gen_waveform,
]


# ── Public API ───────────────────────────────────────────────────────────────

def generate_broll_clip(
    output_path: str,
    duration: float = 3.0,
    width: int = 1080,
    height: int = 1920,
    style_index: int | None = None,
) -> str:
    """
    Generate a single b-roll clip.

    Args:
        output_path: where to write the .mp4
        duration: clip length in seconds
        width, height: resolution
        style_index: if None, picks randomly for variety

    Returns:
        The output_path on success.
    """
    if style_index is None:
        style_index = random.randint(0, len(_GENERATORS) - 1)

    generator = _GENERATORS[style_index % len(_GENERATORS)]
    print(f"[B-Roll] Generating {generator.__name__} ({duration:.1f}s) → {path.basename(output_path)}")
    generator(output_path, duration, width, height)
    return output_path


def generate_broll_set(
    output_dir: str,
    count: int = 3,
    duration: float = 3.0,
    width: int = 1080,
    height: int = 1920,
) -> list[str]:
    """
    Generate a set of diverse b-roll clips.

    Returns list of output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    clips = []
    for i in range(count):
        out = path.join(output_dir, f"broll_{i}.mp4")
        generate_broll_clip(out, duration, width, height, style_index=i)
        clips.append(out)
    return clips
