from llama_api_client import LlamaAPIClient
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from utils.prompt_loader import load_prompt
from utils.rag_helpers import filter_valid_opportunities

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
        docs = self.vectorstore.similarity_search(query, k=k * 4)

        if setasides:
            allowed_setaside = {sa.lower() for sa in setasides}
            docs = [d for d in docs if d.metadata.get("setaside", "").lower() in allowed_setaside]

        if naics_codes:
            allowed_naics = {code.strip() for code in naics_codes}
            docs = [d for d in docs if d.metadata.get("naics") in allowed_naics]

        # Apply filtering based on notice type and deadline
        filtered = []
        for d in docs:
            meta = d.metadata
            if filter_valid_opportunities([
                {
                    "noticeType": meta.get("notice_type"),
                    "responseDeadLine": meta.get("response_deadline"),
                }
            ]):
                filtered.append(d)
            if len(filtered) >= k:
                break

        return filtered[:k]

    def retrieve_context(self, query, k=10, setasides=None, naics_codes=None):
        docs = self.retrieve_docs(query, k=k, setasides=setasides, naics_codes=naics_codes)
        return "\n\n".join([doc.page_content for doc in docs]), docs

    def generate_response(self, query, k=10, setasides=None, naics_codes=None):
        context, _ = self.retrieve_context(query, k=k, setasides=setasides, naics_codes=naics_codes)

        if not context.strip():
            # Avoid sending an empty context to the LLM which would result in
            # hallucinated answers.
            return "No relevant opportunities were found in the vector store."

        prompt = self.prompt_template.format(query=query, context=context)

        completion = self.llm_client.chat.completions.create(
            model="Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ],
        )
        return completion.completion_message.content.text