import os
from pathlib import Path
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma


class ChromaVectorStore:
    """
    Production-grade persistent ChromaDB Vector Store for VidSense AI.
    Features:
    - Persistent SQLite/DuckDB backing (no unsafe pickle deserialization)
    - Metadata filtering by video_id (multi-video ingestion without collision)
    - Relevance score thresholding
    - Granular collection lifecycle management (add, search, delete by video, count)
    """

    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "vidsense_transcripts",
    ):
        self.persist_directory = str(Path(persist_directory).resolve())
        os.makedirs(self.persist_directory, exist_ok=True)
        self.collection_name = collection_name
        self._store: Optional[Chroma] = None

    def get_store(self, embeddings) -> Chroma:
        """Initialize or return the persistent Chroma store instance."""
        if self._store is None:
            self._store = Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings,
                persist_directory=self.persist_directory,
            )
        return self._store

    def build(self, documents: List[Document], embeddings, video_id: Optional[str] = None) -> Chroma:
        """
        Ingests and indexes document chunks into persistent ChromaDB.
        Ensures each chunk has video_id metadata and a deterministic ID for idempotency.
        """
        store = self.get_store(embeddings)

        # Ensure metadata contains video_id and deterministic chunk ids
        ids = []
        for idx, doc in enumerate(documents):
            v_id = doc.metadata.get("video_id") or video_id or "unknown"
            doc.metadata["video_id"] = v_id
            doc.metadata["chunk_index"] = idx
            ids.append(f"{v_id}_chunk_{idx}")

        store.add_documents(documents=documents, ids=ids)
        return store

    def similarity_search(
        self,
        query: str,
        embeddings,
        k: int = 4,
        video_id: Optional[str] = None,
    ) -> List[Document]:
        """
        Search top-k most similar chunks, optionally scoped to a specific video_id.
        """
        store = self.get_store(embeddings)
        filter_dict = {"video_id": video_id} if video_id else None
        return store.similarity_search(query=query, k=k, filter=filter_dict)

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        embeddings,
        k: int = 4,
        video_id: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Search with relevance score metrics for confidence-aware retrieval.
        """
        store = self.get_store(embeddings)
        filter_dict = {"video_id": video_id} if video_id else None
        results = store.similarity_search_with_relevance_scores(
            query=query, k=k, filter=filter_dict
        )
        if score_threshold is not None:
            results = [(doc, score) for doc, score in results if score >= score_threshold]
        return results

    def delete_video(self, video_id: str, embeddings) -> int:
        """Deletes all indexed chunks associated with a specific video_id."""
        store = self.get_store(embeddings)
        try:
            # Query IDs with matching video_id
            collection = store._collection
            existing = collection.get(where={"video_id": video_id})
            ids_to_delete = existing.get("ids", [])
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
            return len(ids_to_delete)
        except Exception:
            return 0

    def count(self, embeddings) -> int:
        """Returns the total number of chunks stored in the collection."""
        store = self.get_store(embeddings)
        return store._collection.count()
