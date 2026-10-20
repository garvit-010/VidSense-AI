import argparse
import re
import sys
from pathlib import Path

# Reconfigure stdout for UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure backend directory is in python path
backend_dir = Path(__file__).resolve().parent
sys.path.append(str(backend_dir))

from app.ingestion.youtube import YouTubeTranscriptLoader
from app.chunking.splitter import DocumentChunker
from app.embeddings.embedder import EmbeddingProvider
from app.vectorstore.chroma_store import ChromaVectorStore


def extract_video_id(url_or_id: str) -> str:
    """
    Extracts 11-character YouTube video ID from various URL formats or raw ID.
    Supports:
      - Raw ID: -VPYbJxPvzY or wjZofJX0v4M
      - Standard URL: https://www.youtube.com/watch?v=-VPYbJxPvzY
      - Short URL: https://youtu.be/-VPYbJxPvzY
      - Embed URL: https://www.youtube.com/embed/-VPYbJxPvzY
    """
    url_or_id = url_or_id.strip()
    if len(url_or_id) == 11 and not re.search(r"[/=&?]", url_or_id):
        return url_or_id

    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"embed\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    return url_or_id


def ingest_video(video_input: str, persist_dir: str = "chroma_db", chunk_size: int = 1000, chunk_overlap: int = 200):
    video_id = extract_video_id(video_input)
    print("=" * 65)
    print("[VIDSENSE INGESTION] YouTube Transcript Ingestion to ChromaDB")
    print("=" * 65)
    print(f"  Target Video ID : {video_id}")
    print(f"  ChromaDB Path   : {persist_dir}")

    # 1. Fetch Transcript
    print("\n[1/3] Fetching transcript via YouTube API...")
    try:
        loader = YouTubeTranscriptLoader(video_id=video_id)
        raw_docs = loader.load()
        full_text_len = len(raw_docs[0].page_content)
        print(f"  [OK] Transcript fetched successfully ({full_text_len:,} characters)")
    except Exception as e:
        print(f"  [ERROR] Failed to fetch transcript for '{video_id}': {e}")
        sys.exit(1)

    # 2. Chunking
    print("\n[2/3] Chunking document text...")
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.split(raw_docs)
    print(f"  [OK] Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")

    # 3. Embedding & ChromaDB Storage
    print("\n[3/3] Generating embeddings & storing in persistent ChromaDB...")
    embedder = EmbeddingProvider()
    embeddings = embedder.get()
    chroma_store = ChromaVectorStore(persist_directory=str(backend_dir / persist_dir))

    chroma_store.build(documents=chunks, embeddings=embeddings, video_id=video_id)
    total_stored = chroma_store.count(embeddings)
    print(f"  [OK] Successfully indexed {len(chunks)} chunks into ChromaDB!")
    print(f"  [OK] Total chunks in collection: {total_stored}")
    print("=" * 65)
    print(f"[SUCCESS] Video '{video_id}' is now indexed and ready for search & evals!")
    print("=" * 65)
    return video_id, len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a YouTube video transcript into ChromaDB")
    parser.add_argument(
        "video_pos",
        nargs="?",
        default=None,
        help="YouTube video ID or URL (positional)",
    )
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        default=None,
        help="YouTube video ID or URL (flag)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size in characters (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap in characters (default: 200)",
    )
    args = parser.parse_args()

    target_video = args.video_pos or args.video or "-VPYbJxPvzY"

    ingest_video(
        video_input=target_video,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
