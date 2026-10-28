# VideoSense: Multimodal Video Question Answering (RAG)

Upload a video, ask questions in natural language, get answers that cite the exact
time ranges, with one-click jump to that moment in the player.

## How it works
1. **Audio**: ffmpeg extracts audio, faster-whisper transcribes it with timestamps.
2. **Visuals**: PySceneDetect finds scenes; scenes are merged/split into 15-60s chunks
   (this is what makes long videos work). Keyframes per chunk are captioned with BLIP.
3. **Fusion**: each chunk becomes one text block (speech + visual captions + time range),
   embedded with MiniLM and stored in FAISS.
4. **Question**: retrieve top-k chunks, pass them to an LLM with a domain-specific prompt,
   answer with `[mm:ss-mm:ss]` citations. The UI shows source frames and jump buttons.

## Setup
```bash
# 1. install ffmpeg (https://ffmpeg.org) and make sure `ffmpeg` is on PATH
pip install -r requirements.txt

# 2a. free/local LLM
ollama pull llama3.2 && ollama serve
# 2b. or Anthropic API
export ANTHROPIC_API_KEY=...   # then pick "anthropic" in the sidebar

streamlit run app.py
# CLI alternative: python qa.py my_video.mp4 "What is being explained at the start?"
```
First run downloads the Whisper, BLIP and MiniLM models. Results are cached per video in `data/`.

## Files
- `config.py`: models, chunk sizes, **domain presets** (general, lecture, medical, traffic, sports)
- `ingest.py`: audio, scenes, keyframes, captions, index building
- `qa.py`: retrieval, prompt, LLM call (Ollama or Anthropic)
- `app.py`: Streamlit UI

## Ideas to extend
- Hybrid retrieval (BM25 + FAISS) and a cross-encoder reranker
- Swap BLIP for a video-LLM (Qwen2-VL, LLaVA-Video) to describe scenes
- Multi-video library search, speaker diarization, evaluation on a Video-QA benchmark
