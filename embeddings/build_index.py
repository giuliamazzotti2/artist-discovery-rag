"""Build ChromaDB embeddings index from artist documents."""

import json
import os
from typing import Dict, List

import chromadb
from chromadb.config import Settings
from rich.console import Console
from sentence_transformers import SentenceTransformer

ROOT_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTISTS_JSON = os.path.join(ROOT_DIRECTORY, "data", "artists.json")
CHROMA_DIR = os.path.join(ROOT_DIRECTORY, "chroma_db")
console = Console()


def load_artist_documents() -> List[Dict]:
    """Load the artist JSON dataset from disk."""
    if not os.path.exists(ARTISTS_JSON):
        raise FileNotFoundError(
            f"Artist dataset not found. Run data/fetch_artists.py first to create {ARTISTS_JSON}."
        )

    with open(ARTISTS_JSON, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def create_embeddings(documents: List[str]) -> List[List[float]]:
    """Create embeddings using all-MiniLM-L6-v2 from sentence-transformers."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    console.print("[blue]Computing embeddings with all-MiniLM-L6-v2...[/blue]")
    embeddings = model.encode(documents, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist()


def build_chroma_index(artists: List[Dict]) -> None:
    """Create or replace a persistent Chroma index for artist documents."""
    if not artists:
        raise ValueError("No artist documents provided to build the index.")

    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        client.delete_collection(name="artists")
    except Exception:
        pass

    collection = client.create_collection(name="artists")
    documents = [artist["document"] for artist in artists]
    metadatas = [
        {
            "name": artist["name"],
            "genres": ", ".join(artist.get("genres", [])),
            "tags": ", ".join(artist.get("tags", [])),
            "similar_artists": ", ".join(artist.get("similar_artists", [])),
        }
        for artist in artists
    ]
    ids = [f"artist-{index}" for index in range(len(artists))]
    embeddings = create_embeddings(documents)

    console.print("[blue]Adding artist documents to ChromaDB index...[/blue]")
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    console.print(
        f"[green]Indexed {len(artists)} artist documents in {CHROMA_DIR}.[/green]"
    )


if __name__ == "__main__":
    artists = load_artist_documents()
    build_chroma_index(artists)
