import argparse
from utils.env_loader import load_env
from agents.solicitation_agent import SolicitationAgent
from chains.semantic_search_chain import SemanticSearchChain
from chains.rerank_chain import RerankChain
from rag.faiss_store import FaissStore

def ingest(config, store):
    agent = SolicitationAgent(config, store)
    agent.run()

def search(store, query):
    search_chain = SemanticSearchChain(store.index)
    results = search_chain.execute(query)
    return results

def rerank(store, query):
    search_chain = SemanticSearchChain(store.index)
    results = search_chain.execute(query)

    rerank_chain = RerankChain()
    top_5 = rerank_chain.execute(query, results)
    print("\n✅ Top Recommended Opportunities:\n")
    print(top_5)

def main():
    parser = argparse.ArgumentParser(description="SAM Solicitation Agent CLI")
    parser.add_argument("--mode", choices=["ingest", "search", "rerank"], required=True, help="Mode to run")
    parser.add_argument("--query", type=str, help="Search query (required for search/rerank)")

    args = parser.parse_args()

    config = load_env()
    store = FaissStore()

    if args.mode == "ingest":
        ingest(config, store)

    elif args.mode == "search":
        if not args.query:
            print("❌ --query is required for search mode.")
            return
        results = search(store, args.query)
        print(f"\n✅ Top {len(results)} Search Results:\n")
        for i, doc in enumerate(results, 1):
            meta = doc.metadata
            print(f"--- [{i}] ---")
            print(f"Title: {meta.get('title')}")
            print(f"Solicitation #: {meta.get('solicitation_number')}")
            print(f"Link: {meta.get('link')}")
            print(f"NAICS: {meta.get('naics')}")
            print(f"Set-Aside: {meta.get('setaside')}")
            print()

    elif args.mode == "rerank":
        if not args.query:
            print("❌ --query is required for rerank mode.")
            return
        rerank(store, args.query)

if __name__ == "__main__":
    main()
