import os
from app.config import HUGGINGFACEHUB_API_TOKEN, GOOGLE_API_KEY


class EmbeddingProvider:
    """
    Production resilient Embedding Provider for ChromaDB vector operations.
    Tiers:
    1. Local SentenceTransformers (all-MiniLM-L6-v2): 100% free, runs locally,
       zero quota limits, immune to HuggingFace 402 Payment Required router errors.
    2. Google Generative AI Embeddings (models/embedding-001) as cloud alternative.
    3. HuggingFace Endpoint Embeddings (if paid router is configured).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._embeddings = None

    def get(self):
        if self._embeddings is not None:
            return self._embeddings

        # 1. Primary: Local sentence-transformers (all-MiniLM-L6-v2)
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            print(f"[EmbeddingProvider] Loading local HuggingFace embeddings: {self.model_name}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            return self._embeddings
        except Exception as e:
            print(f"[EmbeddingProvider] Local HuggingFaceEmbeddings init notice: {e}")

        # 2. Cloud Fallback: Google Generative AI
        if GOOGLE_API_KEY:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                print("[EmbeddingProvider] Falling back to GoogleGenerativeAIEmbeddings.")
                self._embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=GOOGLE_API_KEY,
                )
                return self._embeddings
            except Exception as e:
                print(f"[EmbeddingProvider] Google embeddings init notice: {e}")

        # 3. Cloud Endpoint Fallback
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        self._embeddings = HuggingFaceEndpointEmbeddings(
            repo_id="BAAI/bge-base-en-v1.5",
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        )
        return self._embeddings
