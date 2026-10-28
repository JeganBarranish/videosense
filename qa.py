"""Retrieve relevant video segments and answer with an LLM, citing timestamps."""
import json
import os

import config
from ingest import fmt_time


def load(vid):
    import faiss
    work = os.path.join(config.DATA_DIR, vid)
    index = faiss.read_index(os.path.join(work, "index.faiss"))
    with open(os.path.join(work, "chunks.json")) as f:
        chunks = json.load(f)
    return index, chunks


def retrieve(question, index, chunks, k=None):
    import numpy as np
    from models import get_embedder
    k = min(k or config.TOP_K, len(chunks))
    q = get_embedder().encode([question], normalize_embeddings=True)
    scores, ids = index.search(np.asarray(q, dtype="float32"), k)
    hits = [dict(chunks[i], score=float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]
    return sorted(hits, key=lambda h: h["start"])


def build_prompt(question, hits):
    ctx = "\n\n".join(
        f"[{fmt_time(h['start'])}-{fmt_time(h['end'])}]\n{h['text']}" for h in hits)
    return f"Video context:\n\n{ctx}\n\nQuestion: {question}"


def call_llm(backend, system, user):
    if backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        resp = client.messages.create(
            model=config.ANTHROPIC_MODEL, max_tokens=800, system=system,
            messages=[{"role": "user", "content": user}])
        return resp.content[0].text
    import requests
    r = requests.post(f"{config.OLLAMA_URL}/api/chat", timeout=300, json={
        "model": config.OLLAMA_MODEL, "stream": False,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}]})
    r.raise_for_status()
    return r.json()["message"]["content"]


def answer(question, vid, domain="general", backend=None):
    backend = backend or config.LLM_BACKEND
    index, chunks = load(vid)
    hits = retrieve(question, index, chunks)
    system = config.DOMAINS.get(domain, config.DOMAINS["general"]) + " " + config.RULES
    return call_llm(backend, system, build_prompt(question, hits)), hits


if __name__ == "__main__":  # CLI: python qa.py video.mp4 "question"
    import sys
    import ingest
    vid = ingest.build_index(sys.argv[1], progress=print)
    text, hits = answer(sys.argv[2], vid)
    print("\n" + text)
    for h in hits:
        print(f"  source: {fmt_time(h['start'])}-{fmt_time(h['end'])}")
