from chains.base_chain import BaseChain
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from langchain_core.runnables import RunnableSequence

class RerankChain(BaseChain):
    def __init__(self, config):
        self.llm = Ollama(model="llama3")


        self.prompt = PromptTemplate(
            input_variables=["query", "documents"],
            template=config["RERANK_PROMPT_TEMPLATE"],
        )

        self.chain = self.prompt | self.llm

    def execute(self, query, documents):
        combined_docs = "\n\n".join([doc.page_content for doc in documents])
        print("🧠 Asking LLM to rerank based on AAyeAye qualifications...")

        return self.chain.invoke({"query": query, "documents": combined_docs})
