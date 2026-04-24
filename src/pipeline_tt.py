'''
TikTok-format Paper2Video Pipeline
Produces a vertical (9:16, 1080x1920) ~60-second video suitable for TikTok/Reels/Shorts.

Steps:
    1. (LLM)  Slide generation  — limited to --max_slides slides
    2. (VLM)  Subtitle + cursor prompt generation
    3. TTS -> audio;  WhisperX grounding -> cursor positions
    4. Merge slides + audio + cursor + subtitles
    5. TikTok reformat: pad/blur to 9:16 vertical, speed up to fit --max_duration
'''

import cv2
import pdb
import json
import time
import shutil
import asyncio
import os, sys
import argparse
import subprocess
from os import path
from dotenv import load_dotenv
from pdf2image import convert_from_path

load_dotenv(dotenv_path=path.join(path.dirname(path.dirname(path.abspath(__file__))), '.env'))

from speech_gen import tts_per_slide
from subtitle_render import add_subtitles
from cursor_gen import cursor_gen_per_sentence
from slide_code_gen_select_improvement import latex_code_gen_upgrade
from cursor_render import render_video_with_cursor_from_json
from subtitle_cursor_prompt_gen import subtitle_cursor_gen

from wei_utils import get_agent_config


# ── helpers ──────────────────────────────────────────────────────────────────

def link_latex_proj(src_dir, dst_dir):
    """
    Create a working directory at dst_dir that symlinks every file/subdir
    from src_dir. Generated files (slides.tex, etc.) go into dst_dir without
    duplicating the original assets.
    """
    src_dir = os.path.abspath(src_dir)
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"no such dir: {src_dir}")
    os.makedirs(dst_dir, exist_ok=True)
    for entry in os.listdir(src_dir):
        src_path = os.path.join(src_dir, entry)
        dst_path = os.path.join(dst_dir, entry)
        if not os.path.exists(dst_path):
            os.symlink(src_path, dst_path)


def str2list(s):
    if not s:
        return []
    return [int(x) for x in s.split(',')]


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    return float(result.stdout.strip())


def build_atempo_chain(speed: float) -> str:
    """Build chained atempo filters since each atempo is limited to [0.5, 2.0]."""
    filters = []
    remaining = speed
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    filters.append(f"atempo={remaining:.4f}")
    return ",".join(filters)


def trim_slides_to_budget(slide_image_dir: str, max_slides: int):
    """Keep only the first max_slides slide images; remove the rest."""
    imgs = sorted(
        [f for f in os.listdir(slide_image_dir) if f.endswith('.png')],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    for fname in imgs[max_slides:]:
        os.remove(path.join(slide_image_dir, fname))
    kept = min(len(imgs), max_slides)
    print(f"[TikTok] Using {kept} slides (limit: {max_slides})")
    return kept


def tiktok_reformat(
    input_video: str,
    output_video: str,
    width: int = 1080,
    height: int = 1920,
    max_duration: float = 60.0,
    max_speedup: float = 1.75,
):
    """
    Convert a landscape video to TikTok vertical format (default 1080x1920):
      - Blurred full-screen background
      - Clear original video centered (scaled to fit width)
      - If video exceeds max_duration, speed it up (capped at max_speedup)
    Uses ffmpeg via subprocess.
    """
    duration = get_video_duration(input_video)
    speed = 1.0
    if duration > max_duration:
        speed = min(duration / max_duration, max_speedup)
    print(f"[TikTok] Input duration: {duration:.1f}s, target: {max_duration:.1f}s, speed: {speed:.2f}x")

    blur_amount = 20
    scale_w = width

    # Video filter: reformat to vertical, then speed up if needed
    speed_filter = f",setpts=PTS/{speed:.4f}" if speed > 1.0 else ""
    filtergraph = (
        f"[0:v]split=2[bg_raw][fg_raw];"
        f"[bg_raw]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"boxblur={blur_amount}:5{speed_filter}[bg];"
        f"[fg_raw]scale={scale_w}:-2{speed_filter}[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2[outv]"
    )

    # Audio filter: speed up to match
    if speed > 1.0:
        audio_filter = build_atempo_chain(speed)
        audio_opts = ["-filter:a", audio_filter]
    else:
        audio_opts = []

    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-filter_complex", filtergraph,
        "-map", "[outv]",
        "-map", "0:a",
        *audio_opts,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest",
        output_video,
    ]
    print("[TikTok] Reformatting to vertical:", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        print("[TikTok] ffmpeg stderr:", result.stderr)
        raise RuntimeError("ffmpeg reformat failed")
    final_dur = get_video_duration(output_video)
    print(f"[TikTok] Output written to {output_video} ({final_dur:.1f}s)")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paper2Video TikTok Pipeline')
    parser.add_argument('--result_dir',              type=str,       default='./result/zeyu_tt')
    parser.add_argument('--model_name_t',            type=str,       default='gpt-4.1')
    parser.add_argument('--model_name_v',            type=str,       default='gpt-4.1')
    parser.add_argument('--paper_latex_root',        type=str,       default='./assets/demo/latex_proj')
    parser.add_argument('--ref_img',                 type=str,       default='./assets/demo/zeyu.png')
    parser.add_argument('--ref_audio',               type=str,       default='./assets/demo/zeyu.wav')
    parser.add_argument('--ref_text',                type=str,       default=None)
    parser.add_argument('--gpu_list',                type=str2list,  default="")
    parser.add_argument('--if_tree_search',          type=bool,      default=True)
    parser.add_argument('--beamer_templete_prompt',  type=str,       default=None)
    parser.add_argument('--stage',                   type=str,       default='["0"]')
    # slide+subtitle: 1;  tts+cursor: 2;  merge+tiktok: 3;  all: 0
    parser.add_argument('--max_slides',  type=int,   default=4,
                        help='Max slides to keep (fewer slides = less speedup needed)')
    parser.add_argument('--max_duration', type=float, default=60.0,
                        help='Target duration; video is sped up to fit (max 1.75x)')
    parser.add_argument('--max_speedup', type=float, default=1.5,
                        help='Max playback speedup factor (default 1.5x)')
    parser.add_argument('--tiktok_width',  type=int,  default=1080)
    parser.add_argument('--tiktok_height', type=int,  default=1920)
    args = parser.parse_args()
    stage = json.loads(args.stage)
    print("start", "stage:", stage, "gpu_list:", args.gpu_list)

    cursor_img_path = "./src/cursor_image/red.png"
    os.makedirs(args.result_dir, exist_ok=True)
    agent_config_t = get_agent_config(args.model_name_t)
    agent_config_v = get_agent_config(args.model_name_v)
    # Create a working dir with symlinks to the original latex project —
    # the tex compiler needs figures alongside the .tex, but we avoid copying.
    latex_work_dir = path.join(args.result_dir, path.basename(args.paper_latex_root))
    link_latex_proj(args.paper_latex_root, latex_work_dir)
    args.paper_latex_root = latex_work_dir

    if path.exists(path.join(args.result_dir, "sat.json")):
        with open(path.join(args.result_dir, "sat.json"), 'r') as f:
            time_second = json.load(f)
    else:
        time_second = {}

    if path.exists(path.join(args.result_dir, "token.json")):
        with open(path.join(args.result_dir, "token.json"), 'r') as f:
            token_usage = json.load(f)
    else:
        token_usage = {}

    # ── Step 1: Slide Generation ─────────────────────────────────────────────
    slide_latex_path  = path.join(args.paper_latex_root, "slides.tex")
    slide_image_dir   = path.join(args.result_dir, 'slide_imgs')
    os.makedirs(slide_image_dir, exist_ok=True)

    start_time = time.time()
    if "1" in stage or "0" in stage:
        prompt_path = "./src/prompts/slide_beamer_prompt.txt"
        if args.if_tree_search:
            usage_slide, beamer_path = latex_code_gen_upgrade(
                prompt_path=prompt_path,
                tex_dir=args.paper_latex_root,
                beamer_save_path=slide_latex_path,
                model_config_ll=agent_config_t,
                model_config_vl=agent_config_v,
                beamer_temp_name=args.beamer_templete_prompt,
            )
        else:
            paper_latex_path = path.join(args.paper_latex_root, "main.tex")
            usage_slide, beamer_path = latex_code_gen_upgrade(
                prompt_path=prompt_path,
                tex_dir=args.paper_latex_root,
                tex_path=paper_latex_path,
                beamer_save_path=slide_latex_path,
                model_config=agent_config_t,
            )

        slide_imgs = convert_from_path(beamer_path, dpi=400)
        for i, img in enumerate(slide_imgs):
            img.save(path.join(slide_image_dir, f"{i+1}.png"))

        # ── TikTok: trim to max_slides ──
        trim_slides_to_budget(slide_image_dir, args.max_slides)

        if args.model_name_t not in token_usage:
            token_usage[args.model_name_t] = [usage_slide]
        else:
            token_usage[args.model_name_t].append(usage_slide)
        step1_time = time.time()
        time_second["slide_gen"] = [step1_time - start_time]
        print("Slide Generation", step1_time - start_time)

    # ── Step 2: Subtitle + Cursor Prompt Generation ──────────────────────────
    start_time = time.time()
    subtitle_cursor_save_path = path.join(args.result_dir, 'subtitle_w_cursor.txt')
    cursor_save_path           = path.join(args.result_dir, 'cursor.json')
    speech_save_dir            = path.join(args.result_dir, 'audio')

    if "2" in stage or "0" in stage:
        prompt_path = "./src/prompts/slide_subtitle_cursor_prompt.txt"
        subtitle, usage_subtitle = subtitle_cursor_gen(slide_image_dir, prompt_path, agent_config_v)
        with open(subtitle_cursor_save_path, 'w') as f:
            f.write(subtitle)

        if args.model_name_v not in token_usage:
            token_usage[args.model_name_v] = [usage_subtitle]
        else:
            token_usage[args.model_name_v].append(usage_subtitle)
        step2_time = time.time()
        time_second["subtitle_cursor_prompt_gen"] = [step2_time - start_time]
        print("Subtitle and Cursor Prompt Generation", step2_time - start_time)

        # ── Step 3-1: Speech Generation ──────────────────────────────────────
        tts_per_slide(
            model_type='f5',
            script_path=subtitle_cursor_save_path,
            speech_save_dir=speech_save_dir,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
        )
        step3_1_time = time.time()
        time_second["tts"] = [step3_1_time - step2_time]
        print("Speech Generation", step3_1_time - step2_time)

        # ── Step 3-2: Cursor Generation ───────────────────────────────────────
        os.environ["PYTHONHASHSEED"] = "random"
        cursor_token = cursor_gen_per_sentence(
            script_path=subtitle_cursor_save_path,
            slide_img_dir=slide_image_dir,
            slide_audio_dir=speech_save_dir,
            cursor_save_path=cursor_save_path,
            gpu_list=args.gpu_list,
            model_name=args.model_name_v
        )
        token_usage["cursor"] = cursor_token
        step3_2_time = time.time()
        time_second["cursor_gen"] = [step3_2_time - step3_1_time]
        print("Cursor Generation", step3_2_time - step3_1_time)

    # ── Step 4: Merge + TikTok Reformat ──────────────────────────────────────
    start_time = time.time()
    if "3" in stage or "0" in stage:
        tmp_merge_dir = path.join(args.result_dir, "merge")
        tmp_merge_1   = path.join(args.result_dir, "1_merge.mp4")
        tmp_merge_2   = path.join(args.result_dir, "2_merge.mp4")
        tmp_merge_3   = path.join(args.result_dir, "3_merge.mp4")
        tiktok_out    = path.join(args.result_dir, "tiktok_final.mp4")

        image_size = cv2.imread(path.join(slide_image_dir, '1.png')).shape
        size       = max(image_size[0] // 6, image_size[1] // 6)
        num_slide  = len([f for f in os.listdir(slide_image_dir) if f.endswith('.png')])
        speaker_id = args.ref_img.split("/")[-1].replace(".png", "")
        print(f"[Merge] {num_slide} slides, speaker: {speaker_id}")

        merge_cmd = [
            "./src/1_merage_light.bash",
            slide_image_dir, speech_save_dir, tmp_merge_dir,
            str(num_slide), tmp_merge_1, speaker_id,
        ]
        subprocess.run(merge_cmd, text=True, check=True)

        # Render cursor overlay
        cursor_size = size // 6
        render_video_with_cursor_from_json(
            video_path=tmp_merge_1,
            out_video_path=tmp_merge_2,
            json_path=cursor_save_path,
            cursor_img_path=cursor_img_path,
            transition_duration=0.1,
            cursor_size=cursor_size,
        )

        # Render subtitles — larger font for vertical mobile viewing
        # TikTok subtitle font size: scale relative to tiktok_width
        tt_font_size = args.tiktok_width // 22   # ~49px at 1080p
        add_subtitles(tmp_merge_2, tmp_merge_3, tt_font_size)

        # ── TikTok format: vertical + trim ────────────────────────────────────
        tiktok_reformat(
            input_video=tmp_merge_3,
            output_video=tiktok_out,
            width=args.tiktok_width,
            height=args.tiktok_height,
            max_duration=args.max_duration,
            max_speedup=args.max_speedup,
        )

        step4_time = time.time()
        time_second["merge_and_tiktok"] = [step4_time - start_time]
        print("Merge + TikTok Reformat", step4_time - start_time)
        print(f"\n[Done] TikTok video: {tiktok_out}")

    # ── Save timing + token logs ──────────────────────────────────────────────
    with open(path.join(args.result_dir, "sat.json"),   'w') as f: json.dump(time_second,  f, indent=4)
    with open(path.join(args.result_dir, "token.json"), 'w') as f: json.dump(token_usage,  f, indent=4)
