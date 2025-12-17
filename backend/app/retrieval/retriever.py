class VectorRetriever:
    def __init__(self, vector_store, k: int = 4):
        self.vector_store = vector_store
        self.k = k

    def retrieve(self, query: str):
        return self.vector_store.similarity_search(query, k=self.k)
