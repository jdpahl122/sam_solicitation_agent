from utils.env_loader import load_env
from agents.solicitation_agent import SolicitationAgent
from chains.semantic_search_chain import SemanticSearchChain
from chains.rerank_chain import RerankChain

def main():
    config = load_env()
    agent = SolicitationAgent(config)
    agent.run()

    print("\n🧠 Searching for best federal contract matches...")

    search_chain = SemanticSearchChain(agent.store.index)
    results = search_chain.execute("AI contracting work for a small business")

    rerank_chain = RerankChain()
    top_5 = rerank_chain.execute("AI contracting work for a small business", results)

    print("\n✅ Top Recommended Opportunities:\n")
    print(top_5)

if __name__ == "__main__":
    main()
