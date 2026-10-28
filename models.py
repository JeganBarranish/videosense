"""Lazy, cached model loaders so heavy libraries load only when needed."""
from functools import lru_cache
import config


@lru_cache(maxsize=1)
def get_whisper():
    from faster_whisper import WhisperModel
    return WhisperModel(config.WHISPER_MODEL, compute_type="int8")


@lru_cache(maxsize=1)
def get_captioner():
    from transformers import BlipProcessor, BlipForConditionalGeneration
    proc = BlipProcessor.from_pretrained(config.CAPTION_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(config.CAPTION_MODEL).eval()
    return proc, model


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL)
