import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class FAISSVectorStore:
    def __init__(self, index_path: str = "faiss_index"):
        self.index_path = index_path
        self.store = None

    def build(self, documents: list[Document], embeddings):
        self.store = FAISS.from_documents(documents, embeddings)
        return self.store

    def save(self):
        if self.store is None:
            raise ValueError("Vector store is empty")
        self.store.save_local(self.index_path)

    def load(self, embeddings):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError("FAISS index not found")
        self.store = FAISS.load_local(
            self.index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return self.store
