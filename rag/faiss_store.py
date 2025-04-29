from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings

class FaissStore:
    def __init__(self):
        self.embed_model = OllamaEmbeddings(model="nomic-embed-text")
        self.index = None

    def add_documents(self, docs):
        self.index = FAISS.from_texts(docs, embedding=self.embed_model)

        print(f"✅ Stored {len(docs)} documents into FAISS.")
