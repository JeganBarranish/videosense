import os

DATA_DIR = os.environ.get("VQA_DATA_DIR", "data")

# Models (all run locally by default)
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")            # tiny/base/small/medium
CAPTION_MODEL = os.environ.get("CAPTION_MODEL", "Salesforce/blip-image-captioning-base")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# LLM backend: "ollama" (free, local) or "anthropic" (API key needed)
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# Long-video handling: scenes are merged/split into chunks of this length (seconds)
MIN_CHUNK_SEC = 15
MAX_CHUNK_SEC = 60
FRAMES_PER_CHUNK = 3
TOP_K = 4

# Domain presets: change how the assistant interprets the video
DOMAINS = {
    "general": "You are an assistant that answers questions about a video.",
    "lecture": "You are a teaching assistant answering questions about a lecture video. Explain concepts clearly and mention which part of the lecture covers them.",
    "medical": "You are an assistant describing a medical or surgical video. Be precise, do not give diagnoses or treatment advice, only report what the video shows or says.",
    "traffic": "You are an assistant analysing traffic or CCTV footage. Report vehicles, people, events and their timing factually.",
    "sports": "You are a sports analyst answering questions about a match video. Refer to plays, players and moments of the game.",
}

RULES = (
    "Answer ONLY from the video context provided (speech transcript and visual descriptions). "
    "Cite the time range of every claim in [mm:ss-mm:ss] format. "
    "If the context does not contain the answer, say you could not find it in the video."
)
