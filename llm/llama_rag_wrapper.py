from llama_api_client import LlamaAPIClient
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from utils.prompt_loader import load_prompt

class LlamaRAG:
    def __init__(self, vectorstore_path="vector_store", api_key=None):
        self.llm_client = LlamaAPIClient(api_key=api_key)
        self.embed_model = OllamaEmbeddings(model="nomic-embed-text")
        self.vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings=self.embed_model,
            allow_dangerous_deserialization=True
        )
        self.prompt_template = load_prompt("rag_prompt.txt")

    def retrieve_docs(self, query, k=10, setasides=None, naics_codes=None):
        """Retrieve documents matching the query with optional filters."""
        docs = self.vectorstore.similarity_search(query, k=k * 2)

        if setasides:
            allowed_setaside = {sa.lower() for sa in setasides}
            docs = [d for d in docs if d.metadata.get("setaside", "").lower() in allowed_setaside]

        if naics_codes:
            allowed_naics = {code.strip() for code in naics_codes}
            docs = [d for d in docs if d.metadata.get("naics") in allowed_naics]

        return docs[:k]

    def retrieve_context(self, query, k=10, setasides=None, naics_codes=None):
        docs = self.retrieve_docs(query, k=k, setasides=setasides, naics_codes=naics_codes)
        return "\n\n".join([doc.page_content for doc in docs]), docs

    def generate_response(self, query, k=10, setasides=None, naics_codes=None):
        context, _ = self.retrieve_context(query, k=k, setasides=setasides, naics_codes=naics_codes)
        prompt = self.prompt_template.format(query=query, context=context)

        completion = self.llm_client.chat.completions.create(
            model="Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ],
        )
        return completion.completion_message.content.text