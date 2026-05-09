"""
TikTok Audio Engine — voice-over pacing, background music, and audio ducking.

Handles:
  - Removing silence/pauses from TTS outputs for breathless pacing
  - Concatenating per-slide audio into a single VO track
  - Mixing background music with ducking (music lowers when VO is active)
  - Final audio output
"""

import os
import re
import subprocess
from os import path


def _run_ffmpeg(cmd: list[str], label: str = ""):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[Audio] ffmpeg error ({label}):\n{result.stderr[-500:]}")
        raise RuntimeError(f"Audio engine failed: {label}")


def _get_duration(file_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return float(r.stdout.strip())


# ── Silence removal ─────────────────────────────────────────────────────────

def remove_pauses(audio_path: str, output_path: str, threshold_db: int = -35, min_silence_ms: int = 300) -> str:
    """
    Remove silences longer than min_silence_ms from an audio file.
    Uses ffmpeg's silenceremove filter for breathless, fast-paced TTS.
    """
    _run_ffmpeg([
        "ffmpeg", "-y",
        "-i", audio_path,
        "-af", (
            f"silenceremove=stop_periods=-1"
            f":stop_duration={min_silence_ms / 1000.0}"
            f":stop_threshold={threshold_db}dB"
        ),
        "-c:a", "pcm_s16le",
        output_path,
    ], "remove_pauses")
    return output_path


def remove_pauses_batch(audio_dir: str, output_dir: str) -> list[str]:
    """Remove pauses from all audio files in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    audio_files = sorted(
        [f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3', '.m4a'))],
        key=lambda x: int(re.search(r'\d+', x).group())
    )
    outputs = []
    for fname in audio_files:
        inp = path.join(audio_dir, fname)
        out = path.join(output_dir, fname)
        remove_pauses(inp, out)
        outputs.append(out)
    return outputs


# ── Audio concatenation ─────────────────────────────────────────────────────

def concatenate_audio(audio_files: list[str], output_path: str) -> str:
    """Concatenate a list of audio files into a single track."""
    concat_list = output_path + ".concat.txt"
    with open(concat_list, "w") as f:
        for af in audio_files:
            f.write(f"file '{os.path.abspath(af)}'\n")

    _run_ffmpeg([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_list,
        "-c:a", "pcm_s16le",
        output_path,
    ], "concat_audio")

    os.remove(concat_list)
    return output_path


# ── Background music mixing with ducking ─────────────────────────────────────

def mix_with_music(
    vo_path: str,
    music_path: str,
    output_path: str,
    music_vol_normal: float = 0.25,
    music_vol_ducked: float = 0.08,
) -> str:
    """
    Mix voiceover with background music, applying audio ducking.

    The music volume drops when the VO is active (voice-triggered sidechain).
    Uses ffmpeg's sidechaincompress for automatic ducking.
    """
    vo_dur = _get_duration(vo_path)

    _run_ffmpeg([
        "ffmpeg", "-y",
        "-i", vo_path,
        "-stream_loop", "-1", "-i", music_path,
        "-filter_complex", (
            # Lower music base volume
            f"[1:a]volume={music_vol_normal}[music_quiet];"
            # Sidechain compress: music ducks when VO is present
            f"[music_quiet][0:a]sidechaincompress="
            f"threshold=0.02:ratio=8:attack=5:release=200"
            f":level_in=1:level_sc=1[ducked_music];"
            # Mix VO + ducked music
            f"[0:a][ducked_music]amix=inputs=2:duration=first"
            f":dropout_transition=0:normalize=0[out]"
        ),
        "-map", "[out]",
        "-t", str(vo_dur),
        "-c:a", "aac", "-b:a", "192k",
        output_path,
    ], "mix_music")
    return output_path


def mix_without_music(vo_path: str, output_path: str) -> str:
    """If no music file is available, just convert VO to AAC."""
    _run_ffmpeg([
        "ffmpeg", "-y",
        "-i", vo_path,
        "-c:a", "aac", "-b:a", "192k",
        output_path,
    ], "vo_convert")
    return output_path


# ── Per-segment duration extraction ──────────────────────────────────────────

def get_segment_durations(audio_files: list[str], segments_per_slide: list[int]) -> list[float]:
    """
    Given per-slide audio files and the number of script segments per slide,
    estimate the duration allocated to each segment by dividing the slide's
    audio duration proportionally by segment count.

    Returns a list of durations, one per segment.
    """
    durations = []
    for slide_i, count in enumerate(segments_per_slide):
        if slide_i < len(audio_files):
            slide_dur = _get_duration(audio_files[slide_i])
        else:
            slide_dur = 3.0
        if count <= 0:
            continue
        seg_dur = slide_dur / count
        for _ in range(count):
            durations.append(seg_dur)
    return durations


# ── Final mux: video + audio ────────────────────────────────────────────────

def mux_video_audio(
    video_path: str,
    audio_path: str,
    output_path: str,
    fade_out_duration: float = 1.5,
) -> str:
    """
    Combine a silent video track with the mixed audio track.
    Applies fade-out to both video and audio at the end.
    """
    vid_dur = _get_duration(video_path)
    aud_dur = _get_duration(audio_path)
    final_dur = min(vid_dur, aud_dur)
    fade_start = max(0, final_dur - fade_out_duration)

    _run_ffmpeg([
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-vf", f"fade=t=out:st={fade_start:.3f}:d={fade_out_duration}",
        "-af", f"afade=t=out:st={fade_start:.3f}:d={fade_out_duration}",
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path,
    ], "final_mux")
    return output_path
