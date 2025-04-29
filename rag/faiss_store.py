from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
import os
import pickle

class FaissStore:
    def __init__(self, persist_path="faiss_index"):
        self.embed_model = OllamaEmbeddings(model="nomic-embed-text")
        self.persist_path = persist_path
        self.index = None

        # Load if exists
        if os.path.exists(persist_path):
            with open(os.path.join(persist_path, "faiss_store.pkl"), "rb") as f:
                self.index = pickle.load(f)

    def add_documents(self, docs):
        vectors = self.embed_model.embed_documents(docs)
        if self.index is None:
            self.index = FAISS.from_embeddings(vectors, docs)
        else:
            self.index.add_texts(docs, embeddings=vectors)

        os.makedirs(self.persist_path, exist_ok=True)
        with open(os.path.join(self.persist_path, "faiss_store.pkl"), "wb") as f:
            pickle.dump(self.index, f)
