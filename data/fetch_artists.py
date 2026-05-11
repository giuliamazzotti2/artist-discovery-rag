"""Fetch artist metadata from Last.fm and build structured text documents."""

import json
import os
import re
import time
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter, Retry
from rich.console import Console

LASTFM_API_URL = "https://ws.audioscrobbler.com/2.0/"
GENRES = [
    "electronic",
    "jazz",
    "rock",
    "hip-hop",
    "classical",
    "ambient",
    "metal",
    "pop",
]
TARGET_ARTIST_COUNT = 520
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "artists.json")
console = Console()


def get_env_variable(name: str) -> str:
    """Read a required environment variable and raise a clear error if missing."""
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


def create_session() -> requests.Session:
    """Create a requests session with retry and backoff support."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def clean_html(text: str) -> str:
    """Remove HTML tags from Last.fm biography summaries."""
    text = re.sub(r"<[^>]+>", "", text or "")
    return text.strip()


def fetch_lastfm(session: requests.Session, params: Dict) -> Dict:
    """Fetch JSON from Last.fm with retry handling."""
    api_key = get_env_variable("LASTFM_API_KEY")
    params.update({"api_key": api_key, "format": "json"})

    for attempt in range(1, 5):
        response = session.get(LASTFM_API_URL, params=params, timeout=15)
        if response.status_code == 200:
            return response.json()

        wait = attempt * 2
        console.print(
            f"[yellow]Last.fm request failed (status={response.status_code}). Retrying in {wait}s...[/yellow]"
        )
        time.sleep(wait)

    raise RuntimeError(
        f"Last.fm request failed after retrying: {response.status_code} {response.text}"
    )


def fetch_top_artists_for_genre(session: requests.Session, genre: str, limit: int = 80) -> List[str]:
    """Fetch top artists for a genre tag from Last.fm."""
    data = fetch_lastfm(session, {"method": "tag.gettopartists", "tag": genre, "limit": str(limit)})
    artists = data.get("topartists", {}).get("artist", [])
    return [artist.get("name") for artist in artists if artist.get("name")]


def fetch_artist_info(
    session: requests.Session,
    artist_name: str,
    primary_genre: Optional[str] = None,
) -> Optional[Dict]:
    """Fetch detailed artist info, including bio, tags, and similar artists."""
    try:
        data = fetch_lastfm(
            session,
            {
                "method": "artist.getinfo",
                "artist": artist_name,
                "autocorrect": "1",
            },
        )
    except Exception as error:
        console.print(
            f"[red]Unable to fetch artist info for {artist_name}: {error}[/red]"
        )
        return None

    artist = data.get("artist")
    if not artist:
        return None

    raw_tags = artist.get("tags", {}).get("tag", [])
    tags = [tag.get("name") for tag in raw_tags if tag.get("name")]
    genres = [primary_genre] if primary_genre else []
    genres.extend([tag for tag in tags if tag not in genres])
    if not genres:
        genres = [primary_genre or "unknown"]

    similar_artists = [
        entry.get("name")
        for entry in artist.get("similar", {}).get("artist", [])
        if entry.get("name")
    ]

    bio_text = clean_html(
        artist.get("bio", {}).get("summary")
        or artist.get("bio", {}).get("content")
        or "No biography available."
    )

    return {
        "name": artist.get("name", artist_name),
        "genres": genres,
        "tags": tags,
        "bio": bio_text,
        "similar_artists": similar_artists,
    }


def build_document(artist: Dict) -> str:
    """Create a structured text document for an artist."""
    genres_text = ", ".join(artist.get("genres", [])) or "N/A"
    tags_text = ", ".join(artist.get("tags", [])) or "N/A"
    similar_text = ", ".join(artist.get("similar_artists", [])) or "N/A"

    return (
        f"Name: {artist['name']}\n"
        f"Generi: {genres_text}\n"
        f"Tag: {tags_text}\n"
        f"Bio: {artist['bio']}\n"
        f"Artisti simili: {similar_text}"
    )


def fetch_artists() -> List[Dict]:
    """Fetch a balanced set of artist documents across the target genres."""
    load_dotenv()
    session = create_session()

    artist_names = set()
    artist_documents = []

    console.print("[bold blue]Starting Last.fm artist collection...[/bold blue]")
    for genre in GENRES:
        if len(artist_names) >= TARGET_ARTIST_COUNT:
            break

        console.print(f"[blue]Collecting artists for genre:[/blue] {genre}")
        try:
            names = fetch_top_artists_for_genre(session, genre, limit=90)
        except Exception as err:
            console.print(f"[red]Skipping genre '{genre}': {err}[/red]")
            continue
        for name in names:
            normalized_name = name.lower()
            if normalized_name in artist_names:
                continue

            artist_info = fetch_artist_info(session, name, primary_genre=genre)
            if not artist_info:
                continue

            artist_info["document"] = build_document(artist_info)
            artist_documents.append(artist_info)
            artist_names.add(normalized_name)

            if len(artist_names) >= TARGET_ARTIST_COUNT:
                break

    if len(artist_documents) < 500:
        console.print(
            f"[yellow]Collected {len(artist_documents)} artists so far. Expanding with similar artists...[/yellow]"
        )
        for artist in list(artist_documents):
            if len(artist_names) >= TARGET_ARTIST_COUNT:
                break

            for similar_name in artist.get("similar_artists", []):
                normalized_name = similar_name.lower()
                if normalized_name in artist_names:
                    continue

                artist_info = fetch_artist_info(session, similar_name, primary_genre=artist["genres"][0])
                if not artist_info:
                    continue

                artist_info["document"] = build_document(artist_info)
                artist_documents.append(artist_info)
                artist_names.add(normalized_name)

                if len(artist_names) >= TARGET_ARTIST_COUNT:
                    break

    console.print(
        f"[green]Fetched {len(artist_documents)} unique artist documents.[/green]"
    )

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
        json.dump(artist_documents, output_file, ensure_ascii=False, indent=2)

    console.print(f"[green]Saved artist data to {OUTPUT_FILE}[/green]")
    return artist_documents


if __name__ == "__main__":
    fetch_artists()
