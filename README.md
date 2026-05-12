# Artist Discovery RAG



[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-F97316)](https://www.trychroma.com/)
[![Claude AI](https://img.shields.io/badge/Claude-Haiku-7C3AED)](https://www.anthropic.com/)
[![License](https://img.shields.io/badge/License-MIT-22C55E)](LICENSE)

A music discovery app powered by Retrieval-Augmented Generation. Describe a sound, mood, or genre in natural language and get back 5 relevant artist recommendations — each with a concise explanation of why they match. Built with a local ChromaDB vector index over ~500 Last.fm artist profiles and Claude Haiku for generation.

## Screenshot

![Artist Discovery Demo](demo.gif)

---

## How It Works

```
User Query
    │
    ▼
Sentence Embedding (all-MiniLM-L6-v2)
    │
    ▼
ChromaDB Vector Search  ──►  Top 5 Artist Documents
                                      │
                                      ▼
                             Claude Haiku (via LangChain LCEL)
                                      │
                                      ▼
                              5 Recommendations + Explanations
```

The pipeline is built with **LangChain LCEL** (no legacy chains):

```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | ChatPromptTemplate
    | ChatAnthropic
    | StrOutputParser()
)
```

---

## Tech Stack

- **[Streamlit](https://streamlit.io/)** — UI with 3-state machine (home / loading / results)
- **[LangChain LCEL](https://python.langchain.com/)** — modern chain composition (`langchain-core`, `langchain-anthropic`, `langchain-chroma`)
- **[ChromaDB](https://www.trychroma.com/)** — local persistent vector store
- **[sentence-transformers](https://www.sbert.net/)** — `all-MiniLM-L6-v2` for local embeddings (no API calls)
- **[Claude Haiku](https://www.anthropic.com/)** — fast, cost-efficient generation via Anthropic API
- **[Last.fm API](https://www.last.fm/api)** — source of artist metadata (bio, genres, tags, similar artists)
- **python-dotenv** — environment variable management
- **Rich** — progress output during data fetch and indexing

---

## Getting Started

### Prerequisites

- Python 3.11+
- A free [Last.fm API key](https://www.last.fm/api/account/create)
- An [Anthropic API key](https://console.anthropic.com/)

### Installation

```bash
git clone https://github.com/giuliamazzotti2/artist-discovery-rag.git
cd artist-discovery-rag

python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Configure environment variables

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux
```

Edit `.env` and fill in your keys:

```env
ANTHROPIC_API_KEY=your_anthropic_key_here
LASTFM_API_KEY=your_lastfm_key_here

# Optional: override the default Claude model
# CLAUDE_MODEL_NAME=claude-haiku-4-5-20251001
```

### Build the dataset and index

Run these two scripts once to populate the local vector database:

```bash
# 1. Fetch ~500 artist profiles from Last.fm
python data/fetch_artists.py

# 2. Embed documents and build the ChromaDB index
python embeddings/build_index.py
```

### Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project Structure

```
artist-discovery-rag/
├── app.py                   # Streamlit UI — 3-state: home / loading / results
├── requirements.txt         # Pinned Python dependencies
├── .env.example             # Environment variable template
├── .gitignore
├── LICENSE
├── README.md
│
├── data/
│   ├── fetch_artists.py     # Fetches artist metadata from Last.fm API
│   └── artists.json         # Generated dataset (git-ignored, ~500 artists)
│
├── embeddings/
│   └── build_index.py       # Embeds documents and writes ChromaDB index
│
├── rag/
│   └── chain.py             # LCEL RAG chain: retriever → Claude → response
│
└── chroma_db/               # Persistent vector index (git-ignored, generated)
```

---

## Example Queries

| Query | What you get |
|---|---|
| `dark electronic with haunting vocals, like Portishead` | Trip-hop and post-industrial artists with atmospheric production |
| `jazz fusion with electronic elements, like Thundercat` | Boundary-pushing jazz artists mixing synthesis and improvisation |
| `hypnotic minimal techno from Berlin` | Underground electronic artists in the Berghain sound palette |
| `sad indie folk with raw acoustic guitar` | Confessional singer-songwriters in the Elliott Smith lineage |

---

## Future Improvements

- **Audio features integration** — enrich documents with Spotify audio features (tempo, valence, energy) for better similarity signals
- **Spotify / Apple Music links** — surface artist streaming links directly in the result cards
- **User accounts & saved searches** — persist query history across sessions with a lightweight database
- **Multilingual support** — accept queries in any language via multilingual embedding models (`paraphrase-multilingual-MiniLM-L12-v2`)

---

## License

Released under the [MIT License](LICENSE).
