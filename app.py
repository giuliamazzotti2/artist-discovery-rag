"""Streamlit app for Artist Discovery RAG."""

import os
import html
import re
import time
import threading
import streamlit as st
from dotenv import load_dotenv
from rag.chain import query as rag_query

load_dotenv()

_CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")


def _chroma_index_ready() -> bool:
    """Return True if the ChromaDB index directory exists and is non-empty."""
    return os.path.isdir(_CHROMA_DIR) and bool(os.listdir(_CHROMA_DIR))


def render_md(text: str) -> str:
    """Escape HTML, convert markdown to HTML list items."""
    text = html.escape(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)

    items, current, in_list = [], [], False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if re.match(r'^[-•]\s+', line):
            in_list = True
            items.append(re.sub(r'^[-•]\s+', '', line))
        elif in_list and line:
            items[-1] += ' ' + line
        elif not in_list and line:
            current.append(line)

    if items:
        li_tags = ''.join(f'<li>{item}</li>' for item in items)
        return ''.join(f'<p>{l}</p>' for l in current) + f'<ul class="artist-list">{li_tags}</ul>'

    text = re.sub(r'\n{2,}', '<br>', text)
    return text.replace('\n', '<br>')


st.set_page_config(page_title="Artist Discovery", page_icon="🎵", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&display=swap');

/* ── Reset ─────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body { background: #080808 !important; }

[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stVerticalBlock"],
[data-testid="stMainBlockContainer"],
.stApp, .main, section.main { background: transparent !important; }

* { font-family: 'Space Grotesk', sans-serif !important; }

#MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stSidebar"], [data-testid="collapsedControl"] {
    display: none !important;
    visibility: hidden !important;
}

/* ── Orb animations ─────────────────────────────────────── */
@keyframes float-a {
    0%   { transform: translate(0px, 0px) scale(1); }
    25%  { transform: translate(180px, -120px) scale(1.15); }
    50%  { transform: translate(260px, 80px) scale(0.9); }
    75%  { transform: translate(80px, 200px) scale(1.1); }
    100% { transform: translate(0px, 0px) scale(1); }
}
@keyframes float-b {
    0%   { transform: translate(0px, 0px) scale(1); }
    30%  { transform: translate(-200px, 140px) scale(1.2); }
    60%  { transform: translate(120px, -160px) scale(0.85); }
    100% { transform: translate(0px, 0px) scale(1); }
}
@keyframes float-c {
    0%   { transform: translate(0px, 0px) scale(1); }
    40%  { transform: translate(140px, 160px) scale(1.18); }
    80%  { transform: translate(-100px, -80px) scale(0.92); }
    100% { transform: translate(0px, 0px) scale(1); }
}
@keyframes float-d {
    0%   { transform: translate(0px, 0px) scale(1); }
    35%  { transform: translate(-160px, -120px) scale(1.12); }
    70%  { transform: translate(200px, 80px) scale(0.88); }
    100% { transform: translate(0px, 0px) scale(1); }
}

.orb {
    position: fixed;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
}
.orb-1 {
    width: 560px; height: 560px;
    background: radial-gradient(circle, #7C3AED 0%, #4C1D95 60%, transparent 100%);
    filter: blur(80px); opacity: 0.3;
    top: -160px; left: -120px;
    animation: float-a 9s ease-in-out infinite;
}
.orb-2 {
    width: 420px; height: 420px;
    background: radial-gradient(circle, #FACC15 0%, #B45309 60%, transparent 100%);
    filter: blur(90px); opacity: 0.22;
    top: 35%; right: -100px;
    animation: float-b 11s ease-in-out infinite;
}
.orb-3 {
    width: 380px; height: 380px;
    background: radial-gradient(circle, #A855F7 0%, #6D28D9 60%, transparent 100%);
    filter: blur(75px); opacity: 0.28;
    bottom: 0px; left: 25%;
    animation: float-c 8s ease-in-out infinite;
}
.orb-4 {
    width: 280px; height: 280px;
    background: radial-gradient(circle, #EAB308 0%, #92400E 60%, transparent 100%);
    filter: blur(70px); opacity: 0.2;
    top: 18%; left: 55%;
    animation: float-d 12s ease-in-out infinite;
}
.orb-5 {
    width: 220px; height: 220px;
    background: radial-gradient(circle, #C084FC 0%, #7C3AED 60%, transparent 100%);
    filter: blur(60px); opacity: 0.25;
    bottom: 20%; right: 10%;
    animation: float-a 10s ease-in-out infinite reverse;
}

/* ── Layout ─────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] > .main { position: relative; z-index: 1; }
.block-container {
    max-width: 1080px !important;
    padding-top: 3.5rem !important;
    padding-bottom: 8rem !important;
}

/* ── Hero header ─────────────────────────────────────────── */
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(124, 58, 237, 0.15);
    border: 1px solid rgba(168, 85, 247, 0.35);
    border-radius: 50px;
    padding: 7px 18px;
    font-size: 0.9rem;
    font-weight: 700;
    color: #C084FC;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.6rem;
}
.hero-title {
    font-size: clamp(5rem, 14vw, 10rem);
    font-weight: 800;
    letter-spacing: -6px;
    line-height: 0.95;
    padding-bottom: 0.12em;
    overflow: visible;
    background: linear-gradient(135deg, #FFFFFF 0%, #C084FC 40%, #FACC15 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1.4rem;
}
.hero-sub {
    font-size: 1.55rem;
    font-weight: 500;
    color: #6B7280;
    margin-bottom: 3.2rem;
    line-height: 1.5;
}

/* ── Search input ────────────────────────────────────────── */
[data-testid="stTextInput"] label {
    color: #9CA3AF !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    margin-bottom: 10px !important;
}
[data-testid="stTextInput"] input {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1.5px solid rgba(124, 58, 237, 0.35) !important;
    border-radius: 16px !important;
    color: #0A0A0A !important;
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    padding: 22px 26px !important;
    transition: border-color 0.2s, box-shadow 0.2s, background 0.2s !important;
    backdrop-filter: blur(12px) !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #A855F7 !important;
    background: rgba(255, 255, 255, 0.06) !important;
    box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.12),
                0 0 24px rgba(168, 85, 247, 0.08) !important;
    outline: none !important;
}
[data-testid="stTextInput"] input::placeholder { color: #4B5563 !important; }

/* ── Primary button ──────────────────────────────────────── */
[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #7C3AED 0%, #9333EA 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 22px 48px !important;
    font-weight: 800 !important;
    font-size: 1.35rem !important;
    letter-spacing: 0.05em !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 4px 24px rgba(124, 58, 237, 0.4) !important;
    margin-top: 6px !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-3px) scale(1.01) !important;
    box-shadow: 0 10px 36px rgba(124, 58, 237, 0.55) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) scale(0.99) !important;
}

/* ── Secondary (Nuova ricerca) button ────────────────────── */
.btn-secondary [data-testid="stButton"] > button {
    background: transparent !important;
    border: 1.5px solid rgba(124, 58, 237, 0.35) !important;
    color: #A855F7 !important;
    box-shadow: none !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    padding: 14px 32px !important;
}
.btn-secondary [data-testid="stButton"] > button:hover {
    background: rgba(124, 58, 237, 0.08) !important;
    border-color: #A855F7 !important;
    box-shadow: none !important;
    transform: none !important;
}

/* ── Loading state ───────────────────────────────────────── */
@keyframes pulse-glow {
    0%, 100% { opacity: 0.55; }
    50%       { opacity: 1; }
}
@keyframes fade-up {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
.loading-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 55vh;
    text-align: center;
    animation: fade-up 0.4s ease forwards;
}
.loading-eyebrow {
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #4B5563;
    margin-bottom: 1.2rem;
    animation: pulse-glow 2s ease-in-out infinite;
}
.loading-text {
    font-size: 2rem;
    font-weight: 700;
    color: #E5E7EB;
    letter-spacing: -0.8px;
    line-height: 1.3;
    margin-bottom: 2.4rem;
}
.loading-query {
    background: linear-gradient(135deg, #A855F7, #FACC15);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Progress bar — viola */
[data-testid="stProgressBar"] > div {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 50px !important;
    height: 6px !important;
}
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #7C3AED, #A855F7, #FACC15) !important;
    border-radius: 50px !important;
    transition: width 0.1s linear !important;
}

/* ── Results state ───────────────────────────────────────── */
@keyframes card-in {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.results-mini-header {
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #A855F7;
    margin-bottom: 1.6rem;
}
.result-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 20px;
    padding: 28px 28px 24px;
    margin: 0 0 18px;
    position: relative;
    overflow: hidden;
    animation: card-in 0.45s cubic-bezier(0.22, 1, 0.36, 1) forwards;
    backdrop-filter: blur(8px);
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #7C3AED 0%, #FACC15 100%);
}
.card-label {
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #A855F7;
    margin-bottom: 10px;
}
.card-query {
    font-size: 2.2rem;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 20px;
    letter-spacing: -1.5px;
    line-height: 1.1;
}
.card-answer {
    font-size: 1.15rem;
    font-weight: 500;
    color: #D1D5DB;
    line-height: 1.85;
}

/* ── History cards ───────────────────────────────────────── */
.history-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 22px 24px;
    margin: 12px 0;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, background 0.2s;
}
.history-card:hover {
    border-color: rgba(124, 58, 237, 0.25);
    background: rgba(124, 58, 237, 0.04);
}
.history-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #7C3AED, #FACC15);
}
.history-label {
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #6B7280;
    margin-bottom: 6px;
}
.history-query {
    font-size: 1.4rem;
    font-weight: 800;
    color: #E5E7EB;
    margin-bottom: 14px;
    letter-spacing: -0.8px;
    line-height: 1.15;
}
.history-answer {
    font-size: 1rem;
    font-weight: 500;
    color: #9CA3AF;
    line-height: 1.8;
}

/* ── Section title ───────────────────────────────────────── */
.section-title {
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #4B5563;
    margin: 2rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.06);
}

/* ── Artist list ─────────────────────────────────────────── */
.artist-list {
    list-style: none;
    padding: 0;
    margin: 0;
}
.artist-list li {
    display: flex;
    gap: 12px;
    padding: 12px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    font-size: 1.05rem;
    font-weight: 500;
    color: #D1D5DB;
    line-height: 1.6;
}
.artist-list li:last-child { border-bottom: none; }
.artist-list li::before {
    content: '♪';
    color: #A855F7;
    font-size: 1rem;
    flex-shrink: 0;
    margin-top: 2px;
}
.artist-list li strong { color: #FFFFFF; }

/* ── Alerts ──────────────────────────────────────────────── */
[data-testid="stAlert"] {
    background: rgba(250, 204, 21, 0.07) !important;
    border: 1px solid rgba(250, 204, 21, 0.25) !important;
    border-radius: 12px !important;
    color: #FDE047 !important;
}

/* ── Footer bar ──────────────────────────────────────────── */
.footer-bar {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    z-index: 100;
    background: rgba(8, 8, 8, 0.85);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-top: 1px solid rgba(255, 255, 255, 0.07);
    padding: 12px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    flex-wrap: wrap;
}
.footer-brand {
    font-size: 0.82rem;
    font-weight: 700;
    color: #FFFFFF;
    white-space: nowrap;
}
.footer-brand span { color: #A855F7; }
.footer-stack { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.footer-tag {
    font-size: 0.72rem;
    font-weight: 600;
    color: #4B5563;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 50px;
    padding: 3px 10px;
    letter-spacing: 0.02em;
    white-space: nowrap;
}
.footer-tag:hover { color: #9CA3AF; border-color: rgba(168,85,247,0.3); }

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; background: #080808; }
::-webkit-scrollbar-thumb { background: #3B0E8A; border-radius: 3px; }
::-webkit-scrollbar-track { background: transparent; }
</style>

<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
<div class="orb orb-3"></div>
<div class="orb orb-4"></div>
<div class="orb orb-5"></div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
for key, default in [
    ("app_state", "home"),
    ("history", []),
    ("pending_query", ""),
    ("last_answer", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

state = st.session_state.app_state

# ─────────────────────────────────────────────────────────────
# STATE: home
# ─────────────────────────────────────────────────────────────
if state == "home":
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")):
        st.warning("ANTHROPIC_API_KEY is missing. Add it to your .env file.")
    if not _chroma_index_ready():
        st.warning(
            "ChromaDB index not found. "
            "Run `python data/fetch_artists.py` then `python embeddings/build_index.py` to build it."
        )

    st.markdown("""
<div class="hero-badge">✦ Project by Giulia Mazzotti</div>
<div class="hero-title">Artist Discovery</div>
<div class="hero-sub">Describe the sound you're looking for.<br>I'll find the right artists for you.</div>
""", unsafe_allow_html=True)

    query_text = st.text_input(
        "Describe your sound",
        placeholder="e.g. dark electronic with haunting vocals, like Portishead...",
        key="artist_query",
        label_visibility="visible",
    )
    search_button = st.button("Discover artists →", use_container_width=True)

    if search_button:
        if not query_text.strip():
            st.warning("Please describe the sound or mood you're looking for.")
        else:
            st.session_state.pending_query = query_text
            st.session_state.app_state = "loading"
            st.rerun()

# ─────────────────────────────────────────────────────────────
# STATE: loading
# ─────────────────────────────────────────────────────────────
elif state == "loading":
    query = st.session_state.pending_query
    safe_q = html.escape(query)

    st.markdown(f"""
<div class="loading-wrap">
    <div class="loading-eyebrow">Searching</div>
    <div class="loading-text">
        Finding artists for<br>
        <span class="loading-query">"{safe_q}"</span>
    </div>
</div>
""", unsafe_allow_html=True)

    progress_bar = st.progress(0)

    result: dict = {"answer": None, "error": None, "done": False}

    def _run_query() -> None:
        try:
            result["answer"] = rag_query(query)
        except Exception as exc:
            result["error"] = str(exc)
        result["done"] = True

    thread = threading.Thread(target=_run_query, daemon=True)
    thread.start()

    step = 0
    while not result["done"]:
        if step < 88:
            step += 1
        progress_bar.progress(step)
        time.sleep(0.08)

    progress_bar.progress(100)
    time.sleep(0.25)

    if result["error"]:
        st.session_state.last_answer = f"Error: {result['error']}"
    else:
        st.session_state.last_answer = result["answer"]
        st.session_state.history.insert(0, {
            "query": query,
            "answer": result["answer"],
        })

    st.session_state.app_state = "results"
    st.rerun()

# ─────────────────────────────────────────────────────────────
# STATE: results
# ─────────────────────────────────────────────────────────────
elif state == "results":
    st.markdown('<div class="results-mini-header">✦ Artist Discovery</div>', unsafe_allow_html=True)

    latest_q = st.session_state.pending_query
    latest_a = st.session_state.last_answer

    st.markdown(f"""
<div class="result-card">
    <div class="card-label">Result</div>
    <div class="card-query">{html.escape(latest_q)}</div>
    <div class="card-answer">{render_md(latest_a)}</div>
</div>
""", unsafe_allow_html=True)

    if len(st.session_state.history) > 1:
        st.markdown('<div class="section-title">Previous searches</div>', unsafe_allow_html=True)
        for item in st.session_state.history[1:]:
            st.markdown(f"""
<div class="history-card">
    <div class="history-label">Query</div>
    <div class="history-query">{html.escape(item["query"])}</div>
    <div class="history-label" style="margin-top:4px;">Suggested artists</div>
    <div class="history-answer">{render_md(item["answer"])}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="btn-secondary">', unsafe_allow_html=True)
    if st.button("← New search", use_container_width=True, key="new_search"):
        st.session_state.app_state = "home"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer bar (sempre visibile) ──────────────────────────────
st.markdown("""
<div class="footer-bar">
    <div class="footer-brand">Artist <span>Discovery</span></div>
    <div class="footer-stack">
        <span class="footer-tag">Python 3.11+</span>
        <span class="footer-tag">Streamlit</span>
        <span class="footer-tag">LangChain</span>
        <span class="footer-tag">ChromaDB</span>
        <span class="footer-tag">sentence-transformers</span>
        <span class="footer-tag">Claude Haiku</span>
    </div>
</div>
""", unsafe_allow_html=True)
