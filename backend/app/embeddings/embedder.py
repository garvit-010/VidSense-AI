from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import GOOGLE_API_KEY


class EmbeddingProvider:
    def __init__(self, model: str = "models/gemini-embedding-001"):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=GOOGLE_API_KEY
        )

    def get(self):
        return self.embeddings
