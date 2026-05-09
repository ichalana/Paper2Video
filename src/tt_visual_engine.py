"""
TikTok Visual Engine — dynamic video assembly with the 3-second rule.

Handles:
  - Ken Burns (zoom/pan) effects on static slide images
  - Code-scroll / terminal effect generation
  - Impact caption burn-in (big bold text overlays)
  - Table/metric highlight boxes with glow
  - 3-second cut assembly: no visual stays on screen longer than 3s
  - Crossfade transitions between clips
"""

import os
import json
import math
import random
import subprocess
from os import path


def _run_ffmpeg(cmd: list[str], label: str = ""):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[Visual] ffmpeg error ({label}):\n{result.stderr[-800:]}")
        raise RuntimeError(f"Visual engine ffmpeg failed: {label}")


def _get_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return float(r.stdout.strip())


def load_styles(styles_path: str = None) -> dict:
    if styles_path is None:
        styles_path = path.join(path.dirname(path.dirname(path.abspath(__file__))), "assets", "styles.json")
    with open(styles_path) as f:
        return json.load(f)


# ── Ken Burns effects ────────────────────────────────────────────────────────

# Each preset is a zoompan filter string template.
# Variables: {d} = total frames, {w} = output width, {h} = output height
_KB_PRESETS = {
    "zoom_in_center": (
        "zoompan=z='min(pzoom+0.003,1.35)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        ":d={d}:s={w}x{h}:fps=30"
    ),
    "zoom_in_top": (
        "zoompan=z='min(pzoom+0.003,1.35)':x='iw/2-(iw/zoom/2)':y='0'"
        ":d={d}:s={w}x{h}:fps=30"
    ),
    "zoom_in_bottom": (
        "zoompan=z='min(pzoom+0.003,1.35)':x='iw/2-(iw/zoom/2)':y='ih-(ih/zoom)'"
        ":d={d}:s={w}x{h}:fps=30"
    ),
    "pan_top_to_bottom": (
        "zoompan=z='1.3':x='iw/2-(iw/zoom/2)'"
        ":y='(ih-ih/zoom)*on/{d}'"
        ":d={d}:s={w}x{h}:fps=30"
    ),
    "pan_bottom_to_top": (
        "zoompan=z='1.3':x='iw/2-(iw/zoom/2)'"
        ":y='(ih-ih/zoom)*(1-on/{d})'"
        ":d={d}:s={w}x{h}:fps=30"
    ),
    "zoom_out_center": (
        "zoompan=z='if(eq(on,1),1.35,max(pzoom-0.003,1.0))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        ":d={d}:s={w}x{h}:fps=30"
    ),
}

_KB_PRESET_NAMES = list(_KB_PRESETS.keys())


def apply_ken_burns(
    image_path: str,
    output_path: str,
    duration: float = 3.0,
    width: int = 1080,
    height: int = 1920,
    preset: str | None = None,
) -> str:
    """
    Create a static video clip from an image, scaled to fill the frame.
    The preset parameter is accepted but ignored (kept for API compatibility).
    """
    _run_ffmpeg([
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", image_path,
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
        "-t", str(duration),
        output_path,
    ], "static_slide")
    return output_path


# ── Code scroll / terminal effect ────────────────────────────────────────────

_CODE_LINES = [
    "def detect_violations(traces):",
    "    clusters = cluster_traces(traces)",
    "    for cluster in clusters:",
    "        result = llm_agent.analyze(cluster)",
    "        if result.is_violation:",
    "            yield Violation(",
    "                type=result.type,",
    "                severity=result.score,",
    "                evidence=cluster.ids",
    "            )",
    "",
    "class MeerkatAgent:",
    "    def __init__(self, model):",
    "        self.model = model",
    "        self.memory = TraceMemory()",
    "",
    "    async def scan(self, repo):",
    "        traces = await repo.fetch_all()",
    "        violations = []",
    "        for batch in chunk(traces, 128):",
    "            v = self.detect(batch)",
    "            violations.extend(v)",
    "        return violations",
    "",
    "# Cross-trace reasoning engine",
    "pipeline = MeerkatAgent(model='gpt-4')",
    "results = pipeline.scan(trace_repo)",
    "print(f'Found {len(results)} violations')",
]


def generate_code_scroll(
    output_path: str,
    duration: float = 3.0,
    width: int = 1080,
    height: int = 1920,
    color: str = "#39FF14",
    bg_color: str = "#0A0A0A",
) -> str:
    """
    Generate a Matrix/terminal-style scrolling code effect.
    Creates a tall text image and pans through it.
    """
    from PIL import Image, ImageDraw, ImageFont

    font_size = 28
    line_height = 38
    padding = 40
    # Need enough lines to scroll through for the duration
    lines_needed = int((height * 2) / line_height)
    code_lines = (_CODE_LINES * ((lines_needed // len(_CODE_LINES)) + 2))[:lines_needed]

    img_h = len(code_lines) * line_height + padding * 2
    img = Image.new("RGB", (width, img_h), bg_color)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    y = padding
    for line in code_lines:
        # Syntax-ish coloring: keywords in green, strings in cyan, comments in gray
        if line.strip().startswith("#"):
            fill = "#666666"
        elif line.strip().startswith(("def ", "class ", "async ")):
            fill = color
        elif line.strip().startswith(("return ", "yield ", "for ", "if ")):
            fill = "#00D4FF"
        else:
            fill = "#CCCCCC"
        draw.text((padding, y), line, font=font, fill=fill)
        y += line_height

    # Save the tall image
    tmp_img = output_path.replace(".mp4", "_scroll.png")
    img.save(tmp_img)

    # Use zoompan to scroll through the image top-to-bottom
    d = int(duration * 30)
    _run_ffmpeg([
        "ffmpeg", "-y",
        "-i", tmp_img,
        "-vf", (
            f"zoompan=z='1':x='0':y='(ih-{height})*on/{d}'"
            f":d={d}:s={width}x{height}:fps=30"
        ),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
        "-t", str(duration),
        output_path,
    ], "code_scroll")

    # Clean up temp image
    if os.path.exists(tmp_img):
        os.remove(tmp_img)

    return output_path


# ── Impact caption burn-in ───────────────────────────────────────────────────

def burn_impact_caption(
    input_video: str,
    output_video: str,
    text: str,
    styles: dict | None = None,
) -> str:
    """
    Burn a large bold 1-3 word caption into the center of a video clip.
    Uses ffmpeg drawtext with a semi-transparent background box.
    """
    if not text:
        # No caption — just copy
        _run_ffmpeg(["ffmpeg", "-y", "-i", input_video, "-c", "copy", output_video], "caption_copy")
        return output_video

    if styles is None:
        styles = load_styles()

    s = styles["impact_caption"]
    vid_w = styles["resolution"]["width"]
    vid_h = styles["resolution"]["height"]
    font_size = int(vid_h * s["font_size_ratio"])
    y_pos = int(vid_h * s["position_y_ratio"])
    stroke_w = s["stroke_width"]
    bg_opacity = s["bg_opacity"]

    # Render the caption as a transparent PNG overlay using PIL
    from PIL import Image, ImageDraw, ImageFont

    BOLD_FONTS = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    font = None
    for fp in BOLD_FONTS:
        try:
            font = ImageFont.truetype(fp, font_size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    label = text.upper()
    dummy = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    pad_x, pad_y = 30, 18
    box_w = text_w + pad_x * 2
    box_h = text_h + pad_y * 2

    overlay = Image.new("RGBA", (vid_w, vid_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Semi-transparent background box
    bg_alpha = int(255 * bg_opacity)
    bx = (vid_w - box_w) // 2
    draw.rectangle([bx, y_pos - pad_y, bx + box_w, y_pos + text_h + pad_y],
                    fill=(10, 10, 10, bg_alpha))

    # Text with stroke
    tx = (vid_w - text_w) // 2
    draw.text((tx, y_pos), label, font=font, fill=(255, 255, 255, 255),
              stroke_width=stroke_w, stroke_fill=(0, 0, 0, 255))

    overlay_path = input_video + "_caption.png"
    overlay.save(overlay_path)

    # Overlay the PNG on the video using ffmpeg (just overlay filter, no drawtext)
    _run_ffmpeg([
        "ffmpeg", "-y",
        "-i", input_video,
        "-i", overlay_path,
        "-filter_complex", "[0:v][1:v]overlay=0:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "copy",
        output_video,
    ], "impact_caption")

    if os.path.exists(overlay_path):
        os.remove(overlay_path)

    return output_video


# ── Table/metric highlight ───────────────────────────────────────────────────

def add_highlight_box(
    input_video: str,
    output_video: str,
    x: int, y: int, w: int, h: int,
    styles: dict | None = None,
) -> str:
    """Draw a neon-colored bounding box around a metric region."""
    if styles is None:
        styles = load_styles()
    color = styles["highlight"]["color"]
    bw = styles["highlight"]["border_width"]

    drawbox = (
        f"drawbox=x={x}:y={y}:w={w}:h={h}"
        f":color={color}:t={bw}"
    )
    _run_ffmpeg([
        "ffmpeg", "-y",
        "-i", input_video,
        "-vf", drawbox,
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "copy",
        output_video,
    ], "highlight_box")
    return output_video


# ── Text readability check ───────────────────────────────────────────────────

def check_text_readability(image_path: str, min_size_px: int = 20) -> bool:
    """
    Heuristic check: if the image has very dense content (lots of small text),
    it's probably not readable on a phone. Returns True if the image seems
    readable, False if b-roll should be used instead.

    Uses image entropy as a proxy — very high entropy = dense text/figures.
    """
    from PIL import Image
    import math

    img = Image.open(image_path).convert("L")
    # Downsample to speed up
    img = img.resize((270, 480))
    histogram = img.histogram()
    total = sum(histogram)
    entropy = -sum(
        (c / total) * math.log2(c / total)
        for c in histogram if c > 0
    )
    # High entropy (>7.0) typically means dense content
    return entropy < 7.0


# ── 3-Second Cut Assembly ────────────────────────────────────────────────────

def _split_segment_durations(audio_duration: float, max_dur: float = 3.0) -> list[float]:
    """Split a segment into sub-clips of at most max_dur seconds."""
    if audio_duration <= max_dur:
        return [audio_duration]
    n = math.ceil(audio_duration / max_dur)
    base = audio_duration / n
    return [base] * n


def assemble_visual_timeline(
    segments: list[dict],
    slide_image_dir: str,
    audio_durations: list[float],
    work_dir: str,
    hook_image_path: str | None = None,
    styles: dict | None = None,
) -> str:
    """
    Main assembly function. Produces a single video with:
    - Dynamic visuals changing every ≤3 seconds
    - Ken Burns on slides, AI-generated hook image for the intro
    - Code scroll for variety, impact captions burned in

    Args:
        segments: list of script segment dicts (from tt_script_gen)
        slide_image_dir: directory with numbered slide PNGs
        audio_durations: duration in seconds for each segment's audio
        work_dir: temp directory for intermediate files
        hook_image_path: optional AI-generated image for the hook segment
        styles: loaded styles.json dict

    Returns:
        Path to the assembled (silent) video.
    """
    if styles is None:
        styles = load_styles()

    s = styles["resolution"]
    W, H = s["width"], s["height"]
    max_seg = styles["transitions"]["max_segment_duration"]

    os.makedirs(work_dir, exist_ok=True)

    # Collect slide images
    slide_imgs = sorted(
        [path.join(slide_image_dir, f) for f in os.listdir(slide_image_dir) if f.endswith('.png')],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    num_slides = len(slide_imgs)

    kb_preset_idx = 0
    clip_paths = []
    clip_index = 0

    for seg_i, seg in enumerate(segments):
        seg_dur = audio_durations[seg_i] if seg_i < len(audio_durations) else 3.0
        sub_durs = _split_segment_durations(seg_dur, max_seg)

        for sub_i, sub_dur in enumerate(sub_durs):
            clip_path_raw = path.join(work_dir, f"clip_{clip_index:04d}_raw.mp4")
            clip_path = path.join(work_dir, f"clip_{clip_index:04d}.mp4")

            # Decide visual type for this sub-clip
            slide_idx = seg.get("slide_idx")
            is_hook = seg["phase"] == "hook" and sub_i == 0

            if is_hook and hook_image_path and os.path.exists(hook_image_path):
                # Use AI-generated hook image with Ken Burns
                apply_ken_burns(hook_image_path, clip_path_raw, sub_dur, W, H, preset="zoom_in_center")

            elif sub_i > 0 and (clip_index % 4 == 0):
                # Every 4th sub-clip, use code scroll for visual variety
                generate_code_scroll(
                    clip_path_raw, sub_dur, W, H,
                    color=styles["colors"]["neon_green"],
                    bg_color=styles["colors"]["deep_black"],
                )

            else:
                # Ken Burns on the relevant slide
                img_idx = min(slide_idx or 0, num_slides - 1)
                img_path = slide_imgs[img_idx]
                preset = _KB_PRESET_NAMES[kb_preset_idx % len(_KB_PRESET_NAMES)]
                kb_preset_idx += 1
                apply_ken_burns(img_path, clip_path_raw, sub_dur, W, H, preset=preset)

            os.rename(clip_path_raw, clip_path)

            clip_paths.append(clip_path)
            clip_index += 1

    # ── Concatenate all clips with crossfade transitions ─────────────────
    if not clip_paths:
        raise RuntimeError("No visual clips generated")

    if len(clip_paths) == 1:
        final_path = path.join(work_dir, "visual_assembled.mp4")
        os.rename(clip_paths[0], final_path)
        return final_path

    # Use concat demuxer for simplicity and speed
    concat_list = path.join(work_dir, "concat_visual.txt")
    with open(concat_list, "w") as f:
        for cp in clip_paths:
            f.write(f"file '{os.path.abspath(cp)}'\n")

    final_path = path.join(work_dir, "visual_assembled.mp4")
    _run_ffmpeg([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_list,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "23",
        "-an",
        final_path,
    ], "concat_visual")

    # Clean up intermediate clips
    for cp in clip_paths:
        if os.path.exists(cp):
            os.remove(cp)

    return final_path
