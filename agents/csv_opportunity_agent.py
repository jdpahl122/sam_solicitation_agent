from typing import List, Dict
from tasks.csv_loader_task import CSVLoaderTask
from tasks.csv_preprocess_task import CSVPreprocessTask
from chains.opportunity_matching_chain import OpportunityMatchingChain
from rag.milvus_store import MilvusStore


class CSVOpportunityAgent:
    def __init__(self, csv_file_path: str, store: MilvusStore, model_name: str = "llama3"):
        self.csv_file_path = csv_file_path
        self.store = store
        self.loader_task = CSVLoaderTask(csv_file_path)
        self.preprocess_task = CSVPreprocessTask()
        self.matching_chain = OpportunityMatchingChain(model_name)
    
    def load_and_embed_opportunities(self) -> int:
        """Load CSV data, preprocess it, and embed it in the vector store."""
        
        print("📂 Loading opportunities from CSV...")
        opportunities = self.loader_task.execute()
        
        if not opportunities:
            print("⚠️ No opportunities found in CSV file.")
            return 0
        
        print("🧹 Preprocessing opportunities...")
        processed_docs = self.preprocess_task.execute(opportunities)
        
        if not processed_docs:
            print("⚠️ No processed documents to embed.")
            return 0
        
        print("🧠 Embedding opportunities in vector store...")
        self.store.overwrite_documents(processed_docs)
        
        print(f"✅ Successfully loaded and embedded {len(processed_docs)} opportunities")
        return len(processed_docs)
    
    def find_matching_opportunities(self, company_profile: str, top_k: int = 10) -> List[Dict]:
        """Find and rank opportunities that match the company profile."""
        
        print("🔍 Searching for relevant opportunities...")
        
        # Get all documents from the vector store
        # We'll use a broad search to get many candidates, then rank them with LLM
        broad_search_results = self.store.index.similarity_search(
            company_profile, 
            k=min(100, top_k * 10)  # Get more candidates for better ranking
        )
        
        if not broad_search_results:
            print("⚠️ No opportunities found in vector store.")
            return []
        
        # Convert search results to the expected format
        opportunities = []
        for doc in broad_search_results:
            opportunities.append({
                'text': doc.page_content,
                'metadata': doc.metadata
            })
        
        print(f"📊 Found {len(opportunities)} candidate opportunities")
        
        # Use LLM to rank opportunities
        ranked_opportunities = self.matching_chain.rank_opportunities(
            opportunities, 
            company_profile, 
            top_k
        )
        
        return ranked_opportunities
    
    def run_full_pipeline(self, company_profile: str, top_k: int = 10) -> List[Dict]:
        """Run the complete pipeline: load, embed, and find matching opportunities."""
        
        # Load and embed opportunities
        embedded_count = self.load_and_embed_opportunities()
        
        if embedded_count == 0:
            return []
        
        # Find matching opportunities
        return self.find_matching_opportunities(company_profile, top_k)
    
    def search_existing_opportunities(self, company_profile: str, top_k: int = 10) -> List[Dict]:
        """Search existing embedded opportunities without reloading."""
        return self.find_matching_opportunities(company_profile, top_k) 
