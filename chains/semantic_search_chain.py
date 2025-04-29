from .base_chain import BaseChain

class SemanticSearchChain(BaseChain):
    def __init__(self, faiss_index):
        self.index = faiss_index

    def execute(self, query, k=10):
        print(f"🔎 Performing semantic search for: '{query}'")
        return self.index.similarity_search(query, k=k)
