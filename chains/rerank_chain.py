from chains.base_chain import BaseChain
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from langchain_core.runnables import RunnableSequence
from utils.prompt_loader import load_prompt

class RerankChain(BaseChain):
    def __init__(self):
        self.llm = Ollama(model="llama3")
        prompt_text = load_prompt("rerank_prompt.txt")

        self.prompt = PromptTemplate(
            input_variables=["query", "documents"],
            template=prompt_text
        )

        self.chain = self.prompt | self.llm

    def execute(self, query, documents):
        combined_docs = "\n\n".join([doc.page_content for doc in documents])
        print("🧠 Asking LLM to rerank based on AAyeAye qualifications...")
        return self.chain.invoke({"query": query, "documents": combined_docs})
