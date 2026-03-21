import re
import os
import cv2
import pdb
import json
import string
import subprocess
from os import path
from PIL import Image
from camel.models import ModelFactory
from camel.agents import ChatAgent
from camel.messages import BaseMessage



def draw_red_dots_on_image(image_path, point, radius: int = 5) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    red = (0, 0, 255)
    x, y = int(point[0]), int(point[1])
    cv2.circle(image, (x, y), radius, red, thickness=-1)
    cv2.imwrite("output.jpg", image)

def parse_script(script_text):
    pages = script_text.strip().split("###\n")
    result = []
    for page in pages:
        if not page.strip(): continue
        lines = page.strip().split("\n")
        page_data = []
        for line in lines:
            if "|" not in line: 
                continue
            text, cursor = line.split("|", 1)
            page_data.append([text.strip(), cursor.strip()])
        result.append(page_data)
    return result

def infer_cursor_gemini(instruction, image_path, agent):
    """Use Gemini VLM to find cursor position for the given instruction in the slide image."""
    import time
    ori_image = cv2.imread(image_path)
    orig_h, orig_w = ori_image.shape[:2]

    pil_image = Image.open(image_path)
    prompt = (
        f"This is a presentation slide (size: {orig_w}x{orig_h} pixels). "
        f"Find the location of '{instruction}' in the slide. "
        f"Respond with only pixel coordinates in this exact format: click(x, y) "
        f"where x is the horizontal pixel (0 to {orig_w}) and y is the vertical pixel (0 to {orig_h})."
    )
    message = BaseMessage.make_user_message(
        role_name="user", content=prompt, image_list=[pil_image], meta_dict={}
    )

    for attempt in range(10):
        try:
            response = agent.step(message)
            text = response.msg.content.strip()
            token = prompt + text
            match = re.search(r'click\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)', text)
            if match:
                x = max(0.0, min(float(match.group(1)), float(orig_w)))
                y = max(0.0, min(float(match.group(2)), float(orig_h)))
            else:
                print(f"Could not parse cursor from: {text!r}, using center")
                x, y = float(orig_w // 2), float(orig_h // 2)
            return (x, y), token
        except Exception as e:
            wait = 45
            print(f"Rate limit hit (attempt {attempt+1}/10), waiting {wait}s...")
            time.sleep(wait)

    print(f"All retries exhausted for '{instruction}', using center")
    return (float(orig_w // 2), float(orig_h // 2)), ""

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def get_audio_length(audio_path):
    command = ["ffmpeg", "-i", audio_path]
    result = subprocess.run(command, stderr=subprocess.PIPE, text=True)
    for line in result.stderr.splitlines():
        if "Duration" in line:
            duration_str = line.split("Duration:")[1].split(",")[0].strip()
            hours, minutes, seconds = map(float, duration_str.split(":"))
            return hours * 3600 + minutes * 60 + seconds
    return 0 

def timesteps(subtitles, aligned_result, audio_path):
    aligned_words_in_order = []
    for idx, segment in enumerate(aligned_result["segments"]):
        aligned_words_in_order.extend(segment["words"])
    aligned_words_num = len(aligned_words_in_order) - 1
    
    result = []
    current_idx = 0
    for idx, sentence in enumerate(subtitles):
        words_num = len(re.findall(r'\b\w+\b', sentence.lower()))
        start = aligned_words_in_order[min(aligned_words_num, current_idx)]["end"]
        
        current_idx += words_num
        end = aligned_words_in_order[min(aligned_words_num, current_idx)]["end"]

        duration = {"start": start, "end": end, "text": sentence}
        result.append(duration)
    
    result[0]["start"] = 0
    result[-1]["end"] = get_audio_length(audio_path)
    return result

def cursor_gen_per_sentence(script_path, slide_img_dir, slide_audio_dir, cursor_save_path, gpu_list, model_name="gemini-2.5-flash"):
    with open(script_path, 'r') as f:script_with_cursor = ''.join(f.readlines())
    parsed_speech = parse_script(script_with_cursor)
    cursor_token = ""

    slide_imgs = [name for name in os.listdir(slide_img_dir)]
    slide_imgs = sorted(slide_imgs, key=lambda x: int(re.search(r'\d+', x).group()))
    slide_imgs = [path.join(slide_img_dir, name) for name in slide_imgs]

    ## use VLM API for cursor localization (fast API call, no local model needed)
    from wei_utils import get_agent_config
    agent_config = get_agent_config(model_name)
    model = ModelFactory.create(
        model_platform=agent_config["model_platform"],
        model_type=agent_config["model_type"],
        model_config_dict=agent_config.get("model_config"),
        url=agent_config.get("url", None),
    )
    vlm_agent = ChatAgent(model=model, system_message="You are a helpful assistant that identifies locations of elements in presentation slides.")

    cursor_result = []
    for slide_idx in range(len(parsed_speech)):
        speech_with_cursor = parsed_speech[slide_idx]
        image_path = slide_imgs[slide_idx]
        slide_img = cv2.imread(image_path)
        slide_h_img, slide_w_img = slide_img.shape[:2]
        for sentence_idx, (prompt, cursor_prompt) in enumerate(speech_with_cursor):
            if cursor_prompt.strip().lower() == "no":
                print(f"Skipping cursor for slide {slide_idx}, sentence {sentence_idx} (no cursor)")
                point = (slide_w_img // 2, slide_h_img // 2)
                token = ""
            else:
                print(f"Inferring cursor ({model_name}): slide {slide_idx}, sentence {sentence_idx} — '{cursor_prompt}'")
                point, token = infer_cursor_gemini(cursor_prompt, image_path, vlm_agent)
            cursor_result.append({'slide': slide_idx, 'sentence': sentence_idx, 'speech_text': prompt,
                                   'cursor_prompt': cursor_prompt, 'cursor': point, 'token': token})

    for index in range(len(cursor_result)):
        cursor_token += cursor_result[index]["token"]

    ## timesteps: use WAV duration + word-count proportion (avoids slow WhisperX on CPU)
    import wave as wave_mod
    slide_audio = os.listdir(slide_audio_dir)
    slide_audio = sorted(slide_audio, key=lambda x: int(re.search(r'\d+', x).group()))
    slide_audio = [path.join(slide_audio_dir, name) for name in slide_audio]

    def wav_duration(wav_path):
        with wave_mod.open(wav_path, 'rb') as wf:
            return wf.getnframes() / wf.getframerate()

    global_time = 0.0
    new_slide_sentence_timesteps = []
    for idx, slide_audio_path in enumerate(slide_audio):
        dur = wav_duration(slide_audio_path)
        sentences_this_slide = [info for info in cursor_result if info["slide"] == idx]
        if not sentences_this_slide:
            global_time += dur
            continue
        total_words = sum(max(1, len(re.findall(r'\w+', info["speech_text"]))) for info in sentences_this_slide)
        slide_time = global_time
        for info in sentences_this_slide:
            wc = max(1, len(re.findall(r'\w+', info["speech_text"])))
            seg = dur * wc / total_words
            new_slide_sentence_timesteps.append({
                "start": round(slide_time, 4),
                "end": round(slide_time + seg, 4),
                "text": info["speech_text"],
                "cursor": list(info["cursor"]),
            })
            slide_time += seg
        global_time += dur
    
    with open(cursor_save_path.replace(".json", "_mid.json"), 'w') as f: json.dump(cursor_result, f, indent=2)
    with open(cursor_save_path, 'w') as f: json.dump(new_slide_sentence_timesteps, f, indent=2)
    return len(cursor_token)/4