"""
High-Retention TikTok Paper2Video Pipeline
==========================================

Generates a vertical (9:16, 1080x1920) ~45-second video designed for maximum
retention on TikTok/Reels/Shorts.

Architecture:
  1. Slide Generation  — vertical 9:16 beamer slides (hook-first, visual-heavy)
  2. Script Generation — 3-part structure: Hook (0-3s) / Narrative (3-30s) / Flex (30-45s)
  3. Jargon Filter     — converts academic language to punchy social media style
  4. TTS + Pacing      — breathless voiceover with silence removal
  5. Visual Assembly   — 3-second cut rule, Ken Burns, b-roll, code scroll, impact captions
  6. Audio Mixing      — background music with voice-triggered ducking
  7. Final Mux         — video + audio with fade-out
"""

import cv2
import json
import time
import re
import os
import argparse
from os import path
from dotenv import load_dotenv
from pdf2image import convert_from_path

load_dotenv(dotenv_path=path.join(path.dirname(path.dirname(path.abspath(__file__))), '.env'))

from speech_gen import tts_per_slide
from slide_code_gen_select_improvement import latex_code_gen_upgrade
from wei_utils import get_agent_config

from subtitle_render import add_subtitles
from tt_script_gen import generate_tt_script, segments_to_tts_script
from tt_visual_engine import assemble_visual_timeline, load_styles
from tt_audio_engine import (
    remove_pauses_batch, concatenate_audio, mix_with_music,
    mix_without_music, get_segment_durations, mux_video_audio,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def link_latex_proj(src_dir, dst_dir):
    """Symlink source latex project into a working directory."""
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


def generate_hook_image(segments: list[dict], result_dir: str, width: int = 1080, height: int = 1920) -> str:
    """
    Use OpenAI image generation to create a visually striking hook image
    based on the paper's topic extracted from the script segments.
    """
    from openai import OpenAI

    # Build a prompt from the hook + first narrative segments
    topic_lines = []
    for seg in segments:
        topic_lines.append(seg["text"])
        if len(topic_lines) >= 3:
            break
    topic = " ".join(topic_lines)

    prompt = (
        f"Create a visually striking, futuristic digital illustration for a research paper about: {topic}. "
        f"Style: dark background, neon accents, abstract tech aesthetic. "
        f"NO text, NO words, NO letters, NO human faces. "
        f"Cinematic, high contrast, suitable as a vertical phone wallpaper."
    )

    hook_path = path.join(result_dir, "hook_image.png")
    print(f"[TT] Generating hook image via OpenAI...")
    print(f"[TT] Prompt: {prompt[:120]}...")

    try:
        from dotenv import load_dotenv as _ld
        _ld()

        client = OpenAI()
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1792",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        print(f"[TT] Image URL received, downloading...")

        # Download the image
        import urllib.request
        urllib.request.urlretrieve(image_url, hook_path)

        # Resize to exact target resolution
        img = cv2.imread(hook_path)
        if img is not None:
            resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(hook_path, resized)

        print(f"[TT] Hook image saved: {hook_path}")
        return hook_path

    except Exception as e:
        print(f"[TT] Hook image generation failed: {e}")
        print("[TT] Falling back to first slide for hook visual")
        return ""


def resize_slides(slide_image_dir, width=1080, height=1920):
    """Resize all slide PNGs to exact target resolution."""
    imgs = sorted(
        [f for f in os.listdir(slide_image_dir) if f.endswith('.png')],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    for fname in imgs:
        fpath = path.join(slide_image_dir, fname)
        img = cv2.imread(fpath)
        resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(fpath, resized)
    print(f"[TT] Resized {len(imgs)} slides to {width}x{height}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paper2Video — High-Retention TikTok Pipeline')
    parser.add_argument('--result_dir',              type=str,       default='./result/tiktok_out')
    parser.add_argument('--model_name_t',            type=str,       default='gpt-4.1')
    parser.add_argument('--model_name_v',            type=str,       default='gpt-4.1')
    parser.add_argument('--paper_latex_root',        type=str,       default='./assets/demo/latex_proj')
    parser.add_argument('--ref_audio',               type=str,       default='./assets/demo/zeyu.wav')
    parser.add_argument('--ref_text',                type=str,       default=None)
    parser.add_argument('--gpu_list',                type=str2list,  default="")
    parser.add_argument('--if_tree_search',          type=bool,      default=True)
    parser.add_argument('--beamer_templete_prompt',  type=str,       default=None)
    parser.add_argument('--stage',                   type=str,       default='["0"]')
    # Stages: slide+script: 1 | tts+audio: 2 | visual+final: 3 | all: 0
    parser.add_argument('--tiktok_width',  type=int,  default=2160)
    parser.add_argument('--tiktok_height', type=int,  default=3840)
    parser.add_argument('--music_path',    type=str,   default=None,
                        help='Path to background music file (e.g., assets/carti.wav)')
    args = parser.parse_args()
    stage = json.loads(args.stage)
    print("[TT] Starting high-retention pipeline")
    print(f"     Stages: {stage}, Resolution: {args.tiktok_width}x{args.tiktok_height}")

    os.makedirs(args.result_dir, exist_ok=True)
    styles = load_styles()
    agent_config_t = get_agent_config(args.model_name_t)
    agent_config_v = get_agent_config(args.model_name_v)

    # Set up latex working directory
    latex_work_dir = path.join(args.result_dir, path.basename(args.paper_latex_root))
    link_latex_proj(args.paper_latex_root, latex_work_dir)
    args.paper_latex_root = latex_work_dir

    # Load timing/token logs
    sat_path = path.join(args.result_dir, "sat.json")
    tok_path = path.join(args.result_dir, "token.json")
    time_second = json.load(open(sat_path)) if path.exists(sat_path) else {}
    token_usage = json.load(open(tok_path)) if path.exists(tok_path) else {}

    # Paths
    slide_latex_path = path.join(args.paper_latex_root, "slides.tex")
    slide_image_dir  = path.join(args.result_dir, 'slide_imgs')
    script_save_path = path.join(args.result_dir, 'tt_script.json')
    tts_script_path  = path.join(args.result_dir, 'subtitle_w_cursor.txt')
    speech_save_dir  = path.join(args.result_dir, 'audio')
    paced_audio_dir  = path.join(args.result_dir, 'audio_paced')
    hook_image_path  = path.join(args.result_dir, 'hook_image.png')
    visual_work_dir  = path.join(args.result_dir, 'visual_work')
    tiktok_out       = path.join(args.result_dir, 'tiktok_final.mp4')

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 1: Slide Generation + Script Generation
    # ═══════════════════════════════════════════════════════════════════════
    if "1" in stage or "0" in stage:
        os.makedirs(slide_image_dir, exist_ok=True)

        # ── 1a: Generate vertical beamer slides ──────────────────────────
        t0 = time.time()
        prompt_path = "./src/prompts/slide_beamer_prompt_tt.txt"
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
        resize_slides(slide_image_dir, args.tiktok_width, args.tiktok_height)

        token_usage.setdefault(args.model_name_t, []).append(usage_slide)
        time_second["slide_gen"] = [time.time() - t0]
        print(f"[TT] Slide generation: {time_second['slide_gen'][0]:.1f}s")

        # ── 1b: Generate hook-first TikTok script ────────────────────────
        t0 = time.time()
        segments, raw_script, usage_script = generate_tt_script(slide_image_dir, agent_config_v)

        # Save raw script + parsed segments
        with open(script_save_path, 'w') as f:
            json.dump({
                "raw_script": raw_script,
                "segments": segments,
            }, f, indent=2)

        # Convert to TTS format (###-delimited)
        tts_text = segments_to_tts_script(segments)
        with open(tts_script_path, 'w') as f:
            f.write(tts_text)

        token_usage.setdefault(args.model_name_v, []).append(usage_script)
        time_second["script_gen"] = [time.time() - t0]
        print(f"[TT] Script generation: {time_second['script_gen'][0]:.1f}s")
        print(f"     {len(segments)} segments: "
              f"{sum(1 for s in segments if s['phase']=='hook')} hook / "
              f"{sum(1 for s in segments if s['phase']=='narrative')} narrative / "
              f"{sum(1 for s in segments if s['phase']=='flex')} flex")

        # ── 1c: Generate AI hook image ───────────────────────────────────
        t0 = time.time()
        hook_image_path = generate_hook_image(
            segments, args.result_dir,
            width=args.tiktok_width, height=args.tiktok_height,
        )
        time_second["hook_image_gen"] = [time.time() - t0]
        print(f"[TT] Hook image: {time_second['hook_image_gen'][0]:.1f}s")

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 2: TTS + Audio Pacing + B-Roll Generation
    # ═══════════════════════════════════════════════════════════════════════
    if "2" in stage or "0" in stage:
        # ── 2a: Text-to-Speech ───────────────────────────────────────────
        t0 = time.time()
        tts_per_slide(
            model_type='f5',
            script_path=tts_script_path,
            speech_save_dir=speech_save_dir,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
        )
        time_second["tts"] = [time.time() - t0]
        print(f"[TT] TTS: {time_second['tts'][0]:.1f}s")

        # ── 2b: Remove pauses for breathless pacing ─────────────────────
        t0 = time.time()
        paced_files = remove_pauses_batch(speech_save_dir, paced_audio_dir)
        time_second["pacing"] = [time.time() - t0]
        print(f"[TT] Silence removal: {time_second['pacing'][0]:.1f}s")

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 3: Visual Assembly + Audio Mix + Final Output
    # ═══════════════════════════════════════════════════════════════════════
    if "3" in stage or "0" in stage:
        t0 = time.time()

        # Load segments
        with open(script_save_path) as f:
            script_data = json.load(f)
        segments = script_data["segments"]

        # Collect paced audio files
        paced_files = sorted(
            [path.join(paced_audio_dir, f) for f in os.listdir(paced_audio_dir)
             if f.endswith(('.wav', '.mp3'))],
            key=lambda x: int(re.search(r'\d+', path.basename(x)).group())
        )

        # ── 3a: Calculate per-segment durations from audio ───────────────
        # Count segments per slide
        slide_seg_counts = {}
        for seg in segments:
            key = seg["slide_idx"] if seg["slide_idx"] is not None else -1
            slide_seg_counts[key] = slide_seg_counts.get(key, 0) + 1

        # Map slide indices to audio file indices
        slide_keys = []
        for seg in segments:
            key = seg["slide_idx"] if seg["slide_idx"] is not None else (slide_keys[-1] if slide_keys else 0)
            if key not in slide_keys:
                slide_keys.append(key)

        segments_per_slide = [slide_seg_counts.get(k, slide_seg_counts.get(-1, 1)) for k in slide_keys]
        audio_durations = get_segment_durations(paced_files, segments_per_slide)

        # Pad if needed
        while len(audio_durations) < len(segments):
            audio_durations.append(3.0)

        print(f"[TT] Segment durations: {[f'{d:.1f}s' for d in audio_durations]}")

        # ── 3b: Assemble visual timeline ─────────────────────────────────
        os.makedirs(visual_work_dir, exist_ok=True)
        visual_video = assemble_visual_timeline(
            segments=segments,
            slide_image_dir=slide_image_dir,
            audio_durations=audio_durations,
            work_dir=visual_work_dir,
            hook_image_path=hook_image_path if path.exists(hook_image_path) else None,
            styles=styles,
        )
        print(f"[TT] Visual assembly complete: {visual_video}")

        # ── 3c: Concatenate and mix audio ────────────────────────────────
        vo_concat = path.join(args.result_dir, "vo_concat.wav")
        concatenate_audio(paced_files, vo_concat)

        mixed_audio = path.join(args.result_dir, "mixed_audio.m4a")
        if args.music_path and path.exists(args.music_path):
            mix_with_music(
                vo_path=vo_concat,
                music_path=args.music_path,
                output_path=mixed_audio,
                music_vol_normal=styles["audio"]["music_volume_normal"],
                music_vol_ducked=styles["audio"]["music_volume_ducked"],
            )
            print(f"[TT] Audio mixed with music: {args.music_path}")
        else:
            mix_without_music(vo_concat, mixed_audio)
            if args.music_path:
                print(f"[TT] Warning: music file not found: {args.music_path}, using VO only")
            else:
                print("[TT] No music file specified, using VO only")

        # ── 3d: Final mux — video + audio with fade-out ─────────────────
        muxed_video = path.join(args.result_dir, "muxed.mp4")
        mux_video_audio(
            video_path=visual_video,
            audio_path=mixed_audio,
            output_path=muxed_video,
            fade_out_duration=styles["transitions"]["fade_out_duration"],
        )

        # ── 3e: Render karaoke-style subtitles ──────────────────────────
        tt_font_size = args.tiktok_width // 18  # ~60px at 1080p
        add_subtitles(muxed_video, tiktok_out, tt_font_size)

        time_second["visual_and_mux"] = [time.time() - t0]
        print(f"[TT] Visual + audio + mux + subtitles: {time_second['visual_and_mux'][0]:.1f}s")
        print(f"\n{'='*60}")
        print(f"[DONE] TikTok video: {tiktok_out}")
        print(f"{'='*60}")

    # ── Save timing + token logs ──────────────────────────────────────────
    with open(sat_path, 'w') as f:
        json.dump(time_second, f, indent=4)
    with open(tok_path, 'w') as f:
        json.dump(token_usage, f, indent=4)
