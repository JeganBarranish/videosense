"""Video -> timestamped multimodal chunks -> FAISS index."""
import hashlib
import json
import math
import os
import subprocess

import config


def fmt_time(sec):
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def video_id(path):
    h = hashlib.sha1()
    h.update(str(os.path.getsize(path)).encode())
    with open(path, "rb") as f:
        h.update(f.read(8 * 1024 * 1024))
    return h.hexdigest()[:12]


def video_duration(path):
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps


def extract_audio(video, wav):
    """Returns False if the video has no audio track."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video, "-vn", "-ac", "1", "-ar", "16000", wav],
            check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def transcribe(wav):
    import wave
    import numpy as np
    from models import get_whisper
    with wave.open(wav, "rb") as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = audio.astype(np.float32) / 32768.0
    segments, _ = get_whisper().transcribe(audio, vad_filter=True)
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]

def detect_scenes(video, total):
    from scenedetect import detect, ContentDetector
    scenes = detect(video, ContentDetector())
    out = [(s.get_seconds(), e.get_seconds()) for s, e in scenes]
    return out or [(0.0, total)]


def merge_scenes(scenes, min_sec=None, max_sec=None):
    """Merge tiny scenes and split huge ones so every chunk is min..max seconds."""
    min_sec = min_sec or config.MIN_CHUNK_SEC
    max_sec = max_sec or config.MAX_CHUNK_SEC
    chunks, cur = [], None
    for s, e in scenes:
        cur = [s, e] if cur is None else [cur[0], e]
        if cur[1] - cur[0] >= min_sec:
            chunks.append(cur)
            cur = None
    if cur:
        if chunks:
            chunks[-1][1] = cur[1]
        else:
            chunks.append(cur)
    out = []
    for s, e in chunks:
        n = max(1, math.ceil((e - s) / max_sec))
        step = (e - s) / n
        out += [(s + i * step, s + (i + 1) * step) for i in range(n)]
    return out


def extract_keyframes(video, start, end, n, out_dir, tag):
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(video)
    paths = []
    for i in range(n):
        t = start + (end - start) * (i + 0.5) / n
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        p = os.path.join(out_dir, f"{tag}_{i}.jpg")
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(p)
        paths.append(p)
    cap.release()
    return paths


def caption_frames(paths):
    import torch
    from PIL import Image
    from models import get_captioner
    proc, model = get_captioner()
    out = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        inputs = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=30)
        txt = proc.decode(ids[0], skip_special_tokens=True).strip()
        if txt and txt not in out:
            out.append(txt)
    return out


def speech_for(transcript, start, end):
    return " ".join(
        t["text"] for t in transcript if start <= (t["start"] + t["end"]) / 2 < end
    ).strip()


def build_index(video_path, progress=lambda msg: None):
    import faiss
    import numpy as np
    from models import get_embedder

    vid = video_id(video_path)
    work = os.path.join(config.DATA_DIR, vid)
    ipath = os.path.join(work, "index.faiss")
    cpath = os.path.join(work, "chunks.json")
    if os.path.exists(ipath) and os.path.exists(cpath):
        progress("Loaded cached index")
        return vid

    frames_dir = os.path.join(work, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    total = video_duration(video_path)

    progress("Extracting audio and transcribing speech...")
    wav = os.path.join(work, "audio.wav")
    transcript = transcribe(wav) if extract_audio(video_path, wav) else []

    progress("Detecting scenes...")
    spans = merge_scenes(detect_scenes(video_path, total))

    chunks = []
    for i, (s, e) in enumerate(spans):
        progress(f"Analyzing segment {i + 1}/{len(spans)} ({fmt_time(s)}-{fmt_time(e)})...")
        frames = extract_keyframes(video_path, s, e, config.FRAMES_PER_CHUNK, frames_dir, f"c{i}")
        caps = caption_frames(frames)
        speech = speech_for(transcript, s, e)
        text = f"Speech: {speech or '(none)'}\nVisuals: {'; '.join(caps) or '(none)'}"
        chunks.append({"id": i, "start": s, "end": e, "speech": speech,
                       "visuals": caps, "frames": frames, "text": text})

    progress("Building search index...")
    emb = get_embedder().encode([c["text"] for c in chunks], normalize_embeddings=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(np.asarray(emb, dtype="float32"))
    faiss.write_index(index, ipath)
    with open(cpath, "w") as f:
        json.dump(chunks, f, indent=1)
    return vid
