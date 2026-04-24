
import whisperx as whisper
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip


BOLD_FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _load_font(font_size):
    for fp in BOLD_FONT_PATHS:
        try:
            return ImageFont.truetype(fp, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def _word_groups_from_segments(segments, max_words=4):
    """Group word-level timings into small chunks for display."""
    groups = []
    for seg in segments:
        words = seg.get("words", [])
        if not words:
            words = [{"word": seg["text"].strip(),
                      "start": seg["start"], "end": seg["end"]}]
        # Drop words missing timing
        words = [w for w in words if "start" in w and "end" in w]
        for i in range(0, len(words), max_words):
            chunk = words[i:i + max_words]
            if not chunk:
                continue
            groups.append({
                "words": [w["word"] for w in chunk],
                "starts": [float(w["start"]) for w in chunk],
                "ends": [float(w["end"]) for w in chunk],
            })
    return groups


def _render_group_image(words, active_idx, font_size, video_w):
    """
    Render a single subtitle frame for a word group.
    `active_idx`: index of the word to highlight yellow; -1 means none.
    Returns an RGB numpy array + an alpha mask numpy array.
    """
    font = _load_font(font_size)

    dummy = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy)
    space_w = int(draw.textlength(" ", font=font))
    word_widths = [int(draw.textlength(w, font=font)) for w in words]
    total_w = sum(word_widths) + space_w * (len(words) - 1)

    bbox = draw.textbbox((0, 0), "Hg", font=font)
    text_h = bbox[3] - bbox[1]

    padding_x = 28
    padding_y = 18
    box_w = min(int(total_w + 2 * padding_x), video_w - 20)
    box_h = int(text_h + 2 * padding_y)

    img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 180))
    draw = ImageDraw.Draw(img)

    x = (box_w - total_w) / 2
    y = padding_y - bbox[1]   # compensate for font top-bearing

    for i, word in enumerate(words):
        fill = (255, 255, 0, 255) if i == active_idx else (255, 255, 255, 255)
        draw.text((x, y), word, font=font, fill=fill)
        x += word_widths[i] + space_w

    arr = np.array(img)              # H x W x 4 (RGBA)
    rgb = arr[:, :, :3]              # H x W x 3
    mask = arr[:, :, 3].astype(float) / 255.0
    return rgb, mask


def generate_subtitle_clips(segments, video_w, video_h, font_size):
    """
    Produce one static ImageClip per (word-group, active-word) state.
    Each clip is shown only while that word is being spoken.
    """
    groups = _word_groups_from_segments(segments, max_words=4)
    clips = []
    # Position near the bottom with safe margin
    y_pos = int(video_h - font_size * 2.5)

    for g in groups:
        n = len(g["words"])
        for i in range(n):
            start = g["starts"][i]
            end = g["ends"][i]
            dur = end - start
            if dur <= 0:
                continue
            rgb, mask = _render_group_image(g["words"], i, font_size, video_w)
            clip = (ImageClip(rgb)
                    .set_mask(ImageClip(mask, ismask=True))
                    .set_start(start)
                    .set_duration(dur)
                    .set_position(("center", y_pos)))
            clips.append(clip)
    return clips


def add_subtitles(video_path, output_path, font_size):
    print("[Subtitles] Transcribing with WhisperX (word-level)...")
    model = whisper.load_model("base", device="cpu", compute_type="int8")
    result = model.transcribe(video_path, language="en")

    # Word-level alignment
    model_a, metadata = whisper.load_align_model(
        language_code=result["language"], device="cpu"
    )
    result = whisper.align(
        result["segments"], model_a, metadata, video_path, device="cpu"
    )
    segments = result["segments"]

    print("[Subtitles] Generating karaoke-style subtitle clips...")
    video = VideoFileClip(video_path)
    subs = generate_subtitle_clips(segments, video.w, video.h, font_size)

    print(f"[Subtitles] Rendering {len(subs)} subtitle clips...")
    final = CompositeVideoClip([video] + subs)
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")
