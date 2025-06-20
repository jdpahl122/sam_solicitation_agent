import os
from typing import List, Dict

from langchain_community.vectorstores import Milvus
from langchain_ollama.embeddings import OllamaEmbeddings
from pymilvus import utility, connections


class MilvusStore:
    def __init__(self,
                 host: str = "localhost",
                 port: str = "19530",
                 collection_name: str = "sam_solicitations"):
        self.collection_name = collection_name
        self.embed_model = OllamaEmbeddings(model="nomic-embed-text")
        self.connection_args = {"host": host, "port": port}
        connections.connect(**self.connection_args)
        self.index = Milvus(
            embedding_function=self.embed_model,
            collection_name=self.collection_name,
            connection_args=self.connection_args,
        )

    def add_documents(self, docs_with_metadata: List[Dict]):
        texts = [d["text"] for d in docs_with_metadata]
        metadatas = [d["metadata"] for d in docs_with_metadata]
        self.index.add_texts(texts, metadatas=metadatas)

    def overwrite_documents(self, docs_with_metadata: List[Dict]):
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
        self.index = Milvus.from_texts(
            [d["text"] for d in docs_with_metadata],
            embedding=self.embed_model,
            metadatas=[d["metadata"] for d in docs_with_metadata],
            collection_name=self.collection_name,
            connection_args=self.connection_args,
        )

