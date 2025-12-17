from langchain_huggingface import HuggingFaceEndpointEmbeddings
from app.config import HUGGINGFACEHUB_API_TOKEN

class EmbeddingProvider:
    def __init__(self, repo_id="BAAI/bge-base-en-v1.5"):
        self.embeddings = HuggingFaceEndpointEmbeddings(
            repo_id=repo_id,
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        )


    def get(self):
        return self.embeddings
