import os

import streamlit as st

import config
import ingest
import qa
from ingest import fmt_time

st.set_page_config(page_title="VideoSense", layout="wide")
st.title("VideoSense: ask questions about your video")

with st.sidebar:
    domain = st.selectbox("Domain", list(config.DOMAINS))
    backend = st.selectbox("LLM backend", ["ollama", "anthropic"],
                           index=0 if config.LLM_BACKEND == "ollama" else 1)
    st.caption("ollama needs a running local server; anthropic needs ANTHROPIC_API_KEY.")

st.session_state.setdefault("jump", 0)

up = st.file_uploader("Upload a video", type=["mp4", "mov", "mkv", "avi", "webm"])
if up:
    os.makedirs(os.path.join(config.DATA_DIR, "uploads"), exist_ok=True)
    path = os.path.join(config.DATA_DIR, "uploads", up.name)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(up.getbuffer())
    if st.button("Process video"):
        with st.status("Processing video...", expanded=True) as status:
            vid = ingest.build_index(path, progress=st.write)
            status.update(label="Ready. Ask a question!", state="complete")
        st.session_state.update(vid=vid, path=path, jump=0)
        st.session_state.pop("result", None)

if "vid" in st.session_state:
    left, right = st.columns([3, 2])
    with left:
        st.video(st.session_state.path, start_time=int(st.session_state.jump))
    with right:
        q = st.text_input("Ask a question about the video")
        if st.button("Ask") and q.strip():
            with st.spinner("Searching the video and thinking..."):
                try:
                    text, hits = qa.answer(q, st.session_state.vid, domain, backend)
                    st.session_state.result = (text, hits)
                except Exception as e:
                    st.error(f"LLM call failed: {e}")
        if "result" in st.session_state:
            text, hits = st.session_state.result
            st.markdown(text)
            st.subheader("Sources")
            for h in hits:
                c1, c2 = st.columns([1, 2])
                if h["frames"]:
                    c1.image(h["frames"][0], use_container_width=True)
                c2.write(f"**{fmt_time(h['start'])} - {fmt_time(h['end'])}**")
                if c2.button(f"Jump to {fmt_time(h['start'])}", key=f"jump{h['id']}"):
                    st.session_state.jump = h["start"]
                    st.rerun()
