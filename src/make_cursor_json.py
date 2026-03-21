import re, wave, json, os, cv2

AUDIO_DIR = 'result/demo_output/audio'
SUBTITLE_PATH = 'result/demo_output/subtitle_w_cursor.txt'
CURSOR_JSON_PATH = 'result/demo_output/cursor.json'
SLIDE_IMG_DIR = 'result/demo_output/slide_imgs'

def get_wav_duration(path):
    with wave.open(path, 'rb') as f:
        return f.getnframes() / f.getframerate()

def parse_script(text):
    pages = text.strip().split("###\n")
    result = []
    for page in pages:
        if not page.strip():
            continue
        lines = page.strip().split("\n")
        page_data = []
        for line in lines:
            if "|" not in line:
                continue
            t, c = line.split("|", 1)
            t = t.strip()
            if t.lower() == "no":
                continue
            page_data.append(t)
        result.append(page_data)
    return result

def get_center(idx):
    for name in sorted(os.listdir(SLIDE_IMG_DIR)):
        m = re.search(r'\d+', name)
        if m and int(m.group()) == idx + 1:
            img = cv2.imread(os.path.join(SLIDE_IMG_DIR, name))
            if img is not None:
                h, w = img.shape[:2]
                return [w // 2, h // 2]
    return [960, 540]

with open(SUBTITLE_PATH) as f:
    parsed = parse_script(f.read())

audio_files = sorted(
    [f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')],
    key=lambda x: int(re.search(r'\d+', x).group())
)
durations = [get_wav_duration(os.path.join(AUDIO_DIR, af)) for af in audio_files]

print("Slide durations:", [round(d, 2) for d in durations])

all_entries = []
global_time = 0.0
for slide_idx, sentences in enumerate(parsed):
    dur = durations[slide_idx] if slide_idx < len(durations) else 10.0
    if not sentences:
        global_time += dur
        continue
    cx, cy = get_center(slide_idx)
    total_words = sum(max(1, len(re.findall(r'\w+', t))) for t in sentences)
    slide_time = global_time
    for text in sentences:
        wc = max(1, len(re.findall(r'\w+', text)))
        seg = dur * wc / total_words
        all_entries.append({
            "start": round(slide_time, 4),
            "end": round(slide_time + seg, 4),
            "text": text,
            "cursor": [cx, cy]
        })
        slide_time += seg
    global_time += dur

with open(CURSOR_JSON_PATH, 'w') as f:
    json.dump(all_entries, f, indent=2)

print("Wrote", len(all_entries), "entries. Total:", round(global_time, 1), "s")
