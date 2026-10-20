from typing import List, Optional, Tuple
from langchain_core.documents import Document


class VectorRetriever:
    """
    High-level Vector Retriever supporting both ChromaVectorStore and legacy stores.
    Supports video-level scoping, threshold filtering, and score diagnostics.
    """

    def __init__(self, vector_store, embeddings=None, k: int = 4, score_threshold: Optional[float] = None):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.k = k
        self.score_threshold = score_threshold

    def retrieve(self, query: str, video_id: Optional[str] = None) -> List[Document]:
        """
        Retrieves top-k documents relevant to query, optionally filtered by video_id.
        """
        # If ChromaVectorStore instance
        if hasattr(self.vector_store, "similarity_search") and self.embeddings is not None:
            try:
                return self.vector_store.similarity_search(
                    query=query,
                    embeddings=self.embeddings,
                    k=self.k,
                    video_id=video_id,
                )
            except TypeError:
                pass

        # Standard LangChain VectorStore interface fallback
        if video_id and hasattr(self.vector_store, "similarity_search"):
            try:
                return self.vector_store.similarity_search(
                    query, k=self.k, filter={"video_id": video_id}
                )
            except Exception:
                pass

        return self.vector_store.similarity_search(query, k=self.k)

    def retrieve_with_scores(
        self, query: str, video_id: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieves top-k documents along with relevance scores for agent confidence assessment.
        """
        if hasattr(self.vector_store, "similarity_search_with_relevance_scores") and self.embeddings is not None:
            return self.vector_store.similarity_search_with_relevance_scores(
                query=query,
                embeddings=self.embeddings,
                k=self.k,
                video_id=video_id,
                score_threshold=self.score_threshold,
            )
        elif hasattr(self.vector_store, "similarity_search_with_relevance_scores"):
            return self.vector_store.similarity_search_with_relevance_scores(
                query, k=self.k
            )
        else:
            docs = self.retrieve(query, video_id=video_id)
            return [(d, 1.0) for d in docs]
