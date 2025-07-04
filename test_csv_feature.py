#!/usr/bin/env python3
"""
Test script for the CSV Opportunity Matching feature.

This script demonstrates how to use the new CSV functionality to:
1. Load opportunities from a CSV file
2. Find opportunities that match a company profile using AI evaluation

Prerequisites:
- Ensure you have the ContractOpportunitiesFullCSV.csv file
- Make sure Ollama is running with the llama3 model
- Set up your environment variables in .env
"""

import os
import sys
from utils.env_loader import load_env
from rag.milvus_store import MilvusStore
from agents.csv_opportunity_agent import CSVOpportunityAgent


def test_csv_feature():
    """Test the CSV opportunity matching feature."""
    
    # Example CSV file path - update this to your actual file location
    csv_file_path = "ContractOpportunitiesFullCSV.csv"
    
    # Check if CSV file exists
    if not os.path.exists(csv_file_path):
        print(f"❌ CSV file not found: {csv_file_path}")
        print("Please download ContractOpportunitiesFullCSV.csv and place it in the project root,")
        print("or update the csv_file_path variable in this script.")
        return
    
    # Load environment configuration
    print("🔧 Loading configuration...")
    config = load_env()
    
    # Initialize Milvus store
    print("🗄️ Initializing vector store...")
    store = MilvusStore(
        host=config.get("MILVUS_HOST") or "localhost",
        port=config.get("MILVUS_PORT") or "19530",
        collection_name="csv_opportunities_test"  # Use a test collection
    )
    
    # Initialize CSV agent
    print("🤖 Initializing CSV opportunity agent...")
    csv_agent = CSVOpportunityAgent(csv_file_path, store)
    
    # Example company profile
    company_profile = """
    I'm a Service Disabled Veteran owned small business with a focus in AI, ML, Software Architecture, 
    and VAR opportunities related to software products. I'm mostly looking for set-asides that match 
    my skillset including SDVOSBC set-asides. We specialize in:
    - Artificial Intelligence and Machine Learning solutions
    - Software development and architecture
    - Cloud computing and DevOps
    - Cybersecurity software
    - Data analytics and visualization
    
    We're particularly interested in opportunities with the Department of Veterans Affairs, 
    Department of Defense, and other agencies looking for innovative technology solutions.
    """
    
    print("📂 Loading and embedding opportunities from CSV...")
    print("⏳ This may take a few minutes depending on the CSV size...")
    
    try:
        # Load and embed opportunities
        count = csv_agent.load_and_embed_opportunities()
        
        if count == 0:
            print("❌ No opportunities were loaded from the CSV file")
            return
            
        print(f"✅ Successfully loaded and embedded {count} opportunities")
        
        # Find matching opportunities
        print("\n🎯 Finding opportunities that match the company profile...")
        print("⏳ Evaluating opportunities with AI... this may take a few minutes...")
        
        results = csv_agent.find_matching_opportunities(company_profile, top_k=5)
        
        if results:
            print(f"\n🎯 Top {len(results)} Matching Opportunities:\n")
            print("=" * 80)
            
            for i, result in enumerate(results, 1):
                metadata = result['metadata']
                print(f"\n📋 OPPORTUNITY #{i} - MATCH SCORE: {result['match_score']}/100")
                print("=" * 60)
                print(f"Title: {metadata.get('title', 'N/A')}")
                print(f"Department: {metadata.get('department', 'N/A')}")
                print(f"Set-Aside: {metadata.get('set_aside', 'N/A')}")
                print(f"NAICS Code: {metadata.get('naics_code', 'N/A')}")
                print(f"Award Amount: {metadata.get('award_amount', 'N/A')}")
                print(f"Response Deadline: {metadata.get('response_deadline', 'N/A')}")
                print(f"Link: {metadata.get('link', 'N/A')}")
                
                print(f"\n🤖 AI EVALUATION (abbreviated):")
                print("-" * 40)
                # Print first few lines of evaluation
                eval_lines = result['evaluation'].split('\n')[:10]
                print('\n'.join(eval_lines))
                if len(result['evaluation'].split('\n')) > 10:
                    print("... (truncated for brevity)")
                print("\n" + "=" * 80)
        else:
            print("❌ No matching opportunities found")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Test completed!")
    print("\nTo run the full feature manually, use:")
    print(f"pipenv run python main.py --mode csv-match --csv-file {csv_file_path} --company-profile \"Your company profile here\" --top-k 10")


if __name__ == "__main__":
    print("🧪 Testing CSV Opportunity Matching Feature")
    print("=" * 50)
    test_csv_feature() 
