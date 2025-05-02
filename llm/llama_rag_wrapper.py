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

    def retrieve_context(self, query, k=10):
        docs = self.vectorstore.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])

    def generate_response(self, query):
        context = self.retrieve_context(query)
        prompt = self.prompt_template.format(query=query, context=context)

        completion = self.llm_client.chat.completions.create(
            model="Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ],
        )
        return completion.completion_message.content.text
