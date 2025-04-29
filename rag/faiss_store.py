import os
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings

class FaissStore:
    def __init__(self, persist_dir="vector_store"):
        self.persist_dir = persist_dir
        self.embed_model = OllamaEmbeddings(model="nomic-embed-text")
        self.index = None

        # Load existing index if it exists
        self._load_or_create_index()

    def _load_or_create_index(self):
        if os.path.exists(os.path.join(self.persist_dir, "index.faiss")):
            print(f"📂 Loading FAISS index from '{self.persist_dir}'")
            self.index = FAISS.load_local(
                self.persist_dir,
                embeddings=self.embed_model,
            )
        else:
            print("📄 No existing FAISS index found — starting fresh.")
            self.index = None

    def add_documents(self, docs_with_metadata):
        """
        docs_with_metadata: List[Dict] where each dict contains:
            - 'text': str (content for embedding)
            - 'metadata': dict (title, solicitation number, etc.)
        """
        print(f"🧠 Embedding {len(docs_with_metadata)} documents...")

        texts = [doc['text'] for doc in docs_with_metadata]
        metadatas = [doc['metadata'] for doc in docs_with_metadata]

        if self.index:
            self.index.add_texts(texts, metadatas=metadatas)
        else:
            self.index = FAISS.from_texts(texts, embedding=self.embed_model, metadatas=metadatas)

        print(f"💾 Saving FAISS index to '{self.persist_dir}'")
        self.index.save_local(self.persist_dir)
