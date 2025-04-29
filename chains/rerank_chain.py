from chains.base_chain import BaseChain
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from langchain_core.runnables import RunnableSequence


class RerankChain(BaseChain):
    def __init__(self):
        self.llm = Ollama(model="llama3")

        self.prompt = PromptTemplate(
            input_variables=["query", "documents"],
            template=(
                "You are an expert federal contracting assistant for a Service-Disabled Veteran-Owned Small Business (SDVOSB) in technology.\n\n"
                "Query:\n{query}\n\n"
                "Opportunities:\n{documents}\n\n"
                "Based on relevance and SDVOSB eligibility, rank and return the TOP 5 best-fit solicitations. "
                "For each, return:\n- Title\n- Solicitation Number\n- One-line justification"
            )
        )

        self.chain = self.prompt | self.llm

    def execute(self, query, documents):
        combined_docs = "\n\n".join([doc.page_content for doc in documents])
        print("🧠 Asking LLM to rerank based on AAyeAye qualifications...")

        return self.chain.invoke({"query": query, "documents": combined_docs})
