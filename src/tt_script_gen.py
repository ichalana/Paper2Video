"""
TikTok Script Generator — Hook-first, high-retention scripting engine.

Generates a structured 3-part script (Hook / Narrative / Flex) from slide images,
applies a jargon filter to convert academic language to punchy social-media style,
and outputs segments with impact captions and visual hints.
"""

import re
import os
import json
from os import path
from PIL import Image
from camel.models import ModelFactory
from camel.agents import ChatAgent
from camel.messages import BaseMessage


# ── Jargon filter ────────────────────────────────────────────────────────────

JARGON_MAP = [
    # Sorted longest-first so longer patterns match before shorter substrings
    (r"\bour proposed (?:method|approach|framework|system)\b", "our system"),
    (r"\bwe propose (?:a novel |an? )?", "we built "),
    (r"\bwe introduce (?:a novel |an? )?", "we created "),
    (r"\bwe present (?:a novel |an? )?", "here's "),
    (r"\bour methodology leverages\b", "we built a system that uses"),
    (r"\bour (?:method|approach) leverages\b", "we use"),
    (r"\bleverag(?:es?|ing)\b", "uses"),
    (r"\butiliz(?:es?|ing)\b", "uses"),
    (r"\bdemonstrates? superior performance\b", "beats everything else"),
    (r"\bachieves? state[- ]of[- ]the[- ]art\b", "is the current best"),
    (r"\bstate[- ]of[- ]the[- ]art\b", "best-in-class"),
    (r"\bempirical evaluation\b", "testing"),
    (r"\bexperimental results demonstrate\b", "the results show"),
    (r"\bour experimental findings indicate\b", "we found"),
    (r"\bwe empirically (?:show|demonstrate|validate)\b", "we prove"),
    (r"\bsignificant(?:ly)? outperforms?\b", "crushes"),
    (r"\bsubstantially outperforms?\b", "massively beats"),
    (r"\boutperforms? (?:all )?existing (?:methods|baselines|approaches)\b", "beats every alternative"),
    (r"\boutperforms?\b", "beats"),
    (r"\bnovel (?:approach|method|framework|technique)\b", "new approach"),
    (r"\ba novel\b", "a new"),
    (r"\bin this (?:work|paper|study)\b", "here"),
    (r"\bprior (?:work|art|literature)\b", "previous attempts"),
    (r"\bthe existing literature\b", "what's been tried before"),
    (r"\bfacilitat(?:es?|ing)\b", "enables"),
    (r"\bin the context of\b", "for"),
    (r"\bwith respect to\b", "for"),
    (r"\bin order to\b", "to"),
    (r"\bit is worth noting that\b", "notably"),
    (r"\bit should be noted that\b", ""),
    (r"\brobust(?:ness)? (?:against|to)\b", "resilience to"),
    (r"\bsuperi?or(?:ity)? over\b", "better than"),
]


def apply_jargon_filter(text: str) -> str:
    """Replace academic phrasing with punchy, social-media-friendly alternatives."""
    result = text
    for pattern, replacement in JARGON_MAP:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    # Collapse double spaces created by empty replacements
    result = re.sub(r"  +", " ", result)
    # Fix capitalization after sentence start
    result = re.sub(r"(?<=\. )\w", lambda m: m.group().upper(), result)
    return result.strip()


# ── Script parsing ───────────────────────────────────────────────────────────

def parse_tt_script(raw_script: str) -> list[dict]:
    """
    Parse the structured TikTok script from the LLM into a list of segments.

    Each segment dict has:
      - phase: "hook" | "narrative" | "flex"
      - text: the voiceover line
      - impact_text: 1-3 word impact caption (or None)
      - slide_idx: which slide to reference (0-indexed, or None for b-roll)
    """
    segments = []
    current_phase = "hook"
    current_slide = 0
    slide_pattern = re.compile(r"\[slide[:\s]*(\d+)\]", re.IGNORECASE)

    for line in raw_script.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        # Phase markers
        lower = line.lower()
        if "hook" in lower and line.startswith("==="):
            current_phase = "hook"
            continue
        if "narrative" in lower and line.startswith("==="):
            current_phase = "narrative"
            continue
        if "flex" in lower and line.startswith("==="):
            current_phase = "flex"
            continue
        if line.startswith("===") or line.startswith("---"):
            continue

        # Slide break
        if line.startswith("###"):
            current_slide += 1
            continue

        # Parse line: "script text | IMPACT TEXT" or "script text | visual:broll | IMPACT TEXT"
        parts = [p.strip() for p in line.split("|")]
        if not parts or not parts[0]:
            continue

        text = parts[0]
        impact_text = None
        slide_ref = current_slide

        for part in parts[1:]:
            if part.upper().startswith("IMPACT:"):
                impact_text = part[7:].strip()
            elif part.lower().startswith("visual:broll"):
                slide_ref = None  # signal to use b-roll
            elif part.lower().startswith("visual:slide"):
                sm = slide_pattern.search(part)
                if sm:
                    slide_ref = int(sm.group(1)) - 1  # 1-indexed → 0-indexed

        # If the line has no explicit IMPACT but has | separated parts,
        # the last non-visual part might be the impact text
        if impact_text is None and len(parts) >= 2:
            candidate = parts[-1].strip()
            if not candidate.lower().startswith("visual:") and len(candidate.split()) <= 4:
                impact_text = candidate.upper()

        segments.append({
            "phase": current_phase,
            "text": text,
            "impact_text": impact_text,
            "slide_idx": slide_ref,
        })

    return segments


# ── LLM script generation ───────────────────────────────────────────────────

TT_SCRIPT_PROMPT = """\
You are a viral science communicator creating a TikTok/Reels script from research slides.

RULES:
- Output MUST follow the exact structure below.
- Total script length: 45-55 seconds when read aloud (~130-160 words total). Do NOT exceed 160 words.
- Every line MUST have an impact caption (1-3 punchy words after "| IMPACT:").
- Language must be conversational and urgent. NO academic jargon.
- Use "you" and "we" — talk TO the viewer.
- Numbers and metrics must be emphasized dramatically.
- Keep each line SHORT — under 20 words per line. Punchy beats wordy.

STRUCTURE:

=== HOOK (0-3 seconds — ONE provocative line) ===
A scary question, bold claim, or "did you know" that makes the viewer STOP scrolling. | IMPACT: 1-3 WORDS

=== NARRATIVE (3-30 seconds — the story, 2-3 slide transitions) ===
Line about the problem. Why should the viewer care? | IMPACT: KEYWORD
###
Line about the solution/method — what was built. | IMPACT: KEYWORD
Line about how it works — one key mechanism. | IMPACT: KEYWORD
###
Line about why this approach is different/better. | IMPACT: KEYWORD

=== FLEX (30-50 seconds — the proof + CTA) ===
Line with the most impressive metric. Be dramatic. | IMPACT: NUMBER or RESULT
Line with second metric or real-world finding. | IMPACT: KEYWORD
Final line: call to action (follow, link, comment). | IMPACT: CTA PHRASE

IMPORTANT:
- "###" means a slide transition.
- The HOOK line should NOT reference the paper title or authors.
- NARRATIVE should tell a STORY, not list features.
- FLEX metrics should feel like a mic drop.
"""


def generate_tt_script(slide_image_dir: str, agent_config: dict) -> tuple[list[dict], str, dict]:
    """
    Generate a hook-first TikTok script from slide images.

    Returns:
        segments: list of parsed script segment dicts
        raw_script: the raw LLM output (for saving)
        usage: token usage dict
    """
    model = ModelFactory.create(
        model_platform=agent_config["model_platform"],
        model_type=agent_config["model_type"],
        model_config_dict=agent_config.get("model_config"),
        url=agent_config.get("url", None),
    )
    agent = ChatAgent(model=model, system_message="")

    # Load slide images
    slide_image_list = sorted(
        [path.join(slide_image_dir, f) for f in os.listdir(slide_image_dir) if f.endswith('.png')],
        key=lambda x: int(re.search(r'\d+', path.basename(x)).group())
    )
    images = [Image.open(p) for p in slide_image_list]

    message = BaseMessage.make_user_message(
        role_name="user", content=TT_SCRIPT_PROMPT, image_list=images, meta_dict={}
    )
    response = agent.step(message)
    raw_script = response.msg.content.strip()

    # Apply jargon filter
    filtered_script = apply_jargon_filter(raw_script)

    # Parse into segments
    segments = parse_tt_script(filtered_script)

    return segments, raw_script, response.info.get("usage", {})


def segments_to_tts_script(segments: list[dict]) -> str:
    """
    Convert parsed segments back into the ### delimited format expected by
    the existing TTS pipeline (speech_gen.py).

    Groups segments by slide_idx into pages separated by ###.
    Each line: "text | no"  (cursor is not used in TikTok pipeline).
    """
    pages = {}
    page_order = []
    for seg in segments:
        # Use slide_idx as page key, None (b-roll) gets grouped with previous
        key = seg["slide_idx"] if seg["slide_idx"] is not None else (page_order[-1] if page_order else 0)
        if key not in pages:
            pages[key] = []
            page_order.append(key)
        pages[key].append(seg["text"])

    lines = []
    for i, key in enumerate(page_order):
        for text in pages[key]:
            lines.append(f"{text} | no")
        if i < len(page_order) - 1:
            lines.append("###")

    return "\n".join(lines)
