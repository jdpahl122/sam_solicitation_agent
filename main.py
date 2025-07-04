import argparse
from utils.env_loader import load_env
from utils.solicitation_assets import enrich_record_with_details, parse_s3_path
from agents.solicitation_agent import SolicitationAgent
from agents.csv_opportunity_agent import CSVOpportunityAgent
from chains.semantic_search_chain import SemanticSearchChain
from chains.rerank_chain import RerankChain
from rag.milvus_store import MilvusStore
from llm import LlamaRAG
from scripts.rag_setup import run as run_rag_setup

def ingest(config, store):
    agent = SolicitationAgent(config, store)
    agent.run()

def search(store, query, k=10, setasides=None, naics_codes=None):
    # Check if store has data
    try:
        from pymilvus import utility
        collection_exists = utility.has_collection(store.collection_name)
        
        if not collection_exists:
            print("⚠️ No documents found in vector store. You may need to:")
            print("   1. Run 'csv-load' mode first to load CSV data, or")
            print("   2. Run 'ingest' mode to load SAM.gov data")
            return []
            
    except Exception as e:
        print(f"⚠️ Error checking collection: {e}")
    
    search_chain = SemanticSearchChain(store.index)
    results = search_chain.execute(query, k=k)
    
    if setasides:
        allowed = [sa.lower() for sa in setasides]
        original_count = len(results)
        
        # Use partial matching for set-aside filtering
        def matches_setaside(doc):
            # Check both field names: setaside (SAM.gov) and set_aside (CSV)
            setaside_value = doc.metadata.get("setaside", "") or doc.metadata.get("set_aside", "")
            if not setaside_value:
                return False
            setaside_lower = setaside_value.lower()
            return any(allowed_term in setaside_lower for allowed_term in allowed)
        
        results = [d for d in results if matches_setaside(d)]
        print(f"🔍 After set-aside filtering: {len(results)} results (filtered out {original_count - len(results)})")
    
    if naics_codes:
        allowed_naics = {code.strip() for code in naics_codes}
        original_count = len(results)
        results = [d for d in results if d.metadata.get("naics") in allowed_naics or d.metadata.get("naics_code") in allowed_naics]
        print(f"🔍 After NAICS filtering: {len(results)} results (filtered out {original_count - len(results)})")
    
    return results[:k]

def rerank(store, query):
    search_chain = SemanticSearchChain(store.index)
    results = search_chain.execute(query)

    rerank_chain = RerankChain()
    top_5 = rerank_chain.execute(query, results)
    print("\n✅ Top Recommended Opportunities:\n")
    print(top_5)

def run_rag(query, api_key, setasides=None, naics_codes=None, k=10):
    rag = LlamaRAG("vector_store", api_key=api_key)
    docs = rag.retrieve_docs(query, k=k, setasides=setasides, naics_codes=naics_codes)
    print(f"\n✅ Top {len(docs)} Results:\n")
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        print(f"--- [{i}] ---")
        print(f"Title: {meta.get('title')}")
        print(f"Solicitation #: {meta.get('solicitation_number')}")
        print(f"Posted: {meta.get('posted_date')}")
        print(f"Link: {meta.get('link')}")
        print(f"NAICS: {meta.get('naics')}")
        print(f"Set-Aside: {meta.get('setaside')}")
        print()

    response = rag.generate_response(query, k=k, setasides=setasides, naics_codes=naics_codes)
    print("\n📄 RAG-Enhanced Response:\n")
    print(response)

def search_aayeaye_capabilities(store, k=10):
    """Search for opportunities matching AAyeAye LLC's capabilities statement."""
    print("🔍 Searching for opportunities matching AAyeAye LLC capabilities...")
    
    # AAyeAye LLC NAICS codes and capabilities
    naics_codes = ["541715", "541511", "541712", "541519", "541690"]
    
    # Technical capabilities query
    capabilities_query = (
        "machine learning MLOps artificial intelligence software development "
        "data engineering computer systems design scientific consulting research development "
        "infrastructure automation streaming data pipelines model monitoring "
        "kubernetes cloud native platforms"
    )
    
    print(f"🎯 NAICS Codes: {', '.join(naics_codes)}")
    print(f"🔍 Technical Query: {capabilities_query}")
    
    # Search for small business opportunities
    print("\n--- SMALL BUSINESS OPPORTUNITIES ---")
    sb_results = search(store, capabilities_query, k=k, setasides=["small business"], naics_codes=naics_codes)
    
    if sb_results:
        print(f"✅ Found {len(sb_results)} Small Business opportunities:")
        for i, doc in enumerate(sb_results, 1):
            meta = doc.metadata
            print(f"--- [{i}] ---")
            print(f"Title: {meta.get('title')}")
            print(f"Solicitation #: {meta.get('solicitation_number') or meta.get('sol_number')}")
            print(f"NAICS: {meta.get('naics') or meta.get('naics_code')}")
            print(f"Set-Aside: {meta.get('setaside') or meta.get('set_aside')}")
            print(f"Department: {meta.get('department')}")
            print(f"Link: {meta.get('link')}")
            print()
    else:
        print("❌ No Small Business opportunities found")
    
    # Search for SDVOSB opportunities (for when certification is complete)
    print("\n--- SDVOSB OPPORTUNITIES (For Future Reference) ---")
    sdvosb_results = search(store, capabilities_query, k=k, setasides=["veteran"], naics_codes=naics_codes)
    
    if sdvosb_results:
        print(f"✅ Found {len(sdvosb_results)} SDVOSB opportunities:")
        for i, doc in enumerate(sdvosb_results, 1):
            meta = doc.metadata
            print(f"--- [{i}] ---")
            print(f"Title: {meta.get('title')}")
            print(f"Solicitation #: {meta.get('solicitation_number') or meta.get('sol_number')}")
            print(f"NAICS: {meta.get('naics') or meta.get('naics_code')}")
            print(f"Set-Aside: {meta.get('setaside') or meta.get('set_aside')}")
            print(f"Department: {meta.get('department')}")
            print(f"Link: {meta.get('link')}")
            print()
    else:
        print("❌ No SDVOSB opportunities found")
    
    # Search for unrestricted opportunities in your NAICS codes
    print("\n--- UNRESTRICTED OPPORTUNITIES ---")
    unrestricted_results = search(store, capabilities_query, k=k, naics_codes=naics_codes)
    
    if unrestricted_results:
        print(f"✅ Found {len(unrestricted_results)} total opportunities in your NAICS codes:")
        for i, doc in enumerate(unrestricted_results, 1):
            meta = doc.metadata
            set_aside = meta.get('setaside') or meta.get('set_aside') or "None"
            print(f"--- [{i}] ---")
            print(f"Title: {meta.get('title')}")
            print(f"Solicitation #: {meta.get('solicitation_number') or meta.get('sol_number')}")
            print(f"NAICS: {meta.get('naics') or meta.get('naics_code')}")
            print(f"Set-Aside: {set_aside}")
            print(f"Department: {meta.get('department')}")
            print(f"Link: {meta.get('link')}")
            print()
    else:
        print("❌ No opportunities found in your NAICS codes")

def main():
    parser = argparse.ArgumentParser(description="SAM Solicitation Agent CLI")
    parser.add_argument(
        "--mode",
        choices=["ingest", "search", "rerank", "rag", "enrich", "ragsetup", "csv-load", "csv-match", "aayeaye"],
        required=True,
        help="Mode to run",
    )
    parser.add_argument("--query", type=str, help="Search query (required for search/rerank/rag)")
    parser.add_argument(
        "--setaside",
        type=str,
        help="Comma-separated list of set-asides to filter results",
    )
    parser.add_argument(
        "--naics",
        type=str,
        help="Comma-separated list of NAICS codes to filter results",
    )
    parser.add_argument(
        "--path",
        type=str,
        help="S3 path for a single record, e.g. sam-archive/2025/06/16/<id>.json",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Enrich every JSON record in the specified bucket",
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Process only records from this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="sam-archive",
        help="Bucket to use with --all or when --path lacks a bucket",
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        help="Path to ContractOpportunitiesFullCSV.csv file (required for csv-load and csv-match modes)",
    )
    parser.add_argument(
        "--company-profile",
        type=str,
        help="Company profile description for opportunity matching (required for csv-match mode)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top opportunities to return (default: 10)",
    )

    args = parser.parse_args()
    setaside_list = None
    if args.setaside:
        setaside_list = [s.strip() for s in args.setaside.split(',') if s.strip()]
    naics_list = None
    if args.naics:
        naics_list = [c.strip() for c in args.naics.split(',') if c.strip()]

    config = load_env()
    store = MilvusStore(
        host=config.get("MILVUS_HOST") or "localhost",
        port=config.get("MILVUS_PORT") or "19530",
    )

    if args.mode == "ingest":
        ingest(config, store)

    elif args.mode == "search":
        if not args.query:
            print("❌ --query is required for search mode.")
            return
        results = search(store, args.query, k=args.top_k, setasides=setaside_list, naics_codes=naics_list)
        print(f"\n✅ Top {len(results)} Search Results:\n")
        
        if not results:
            print("❌ No results found. Try:")
            print("   • Using different search terms")
            print("   • Removing set-aside or NAICS filters")
            print("   • Running csv-load mode first if searching CSV data")
            print("   • Running ingest mode first if searching SAM.gov data")
        
        for i, doc in enumerate(results, 1):
            meta = doc.metadata
            print(f"--- [{i}] ---")
            print(f"Title: {meta.get('title')}")
            print(f"Solicitation #: {meta.get('solicitation_number') or meta.get('sol_number')}")
            print(f"Posted: {meta.get('posted_date')}")
            print(f"Link: {meta.get('link')}")
            print(f"NAICS: {meta.get('naics') or meta.get('naics_code')}")
            print(f"Set-Aside: {meta.get('setaside') or meta.get('set_aside')}")
            print(f"Department: {meta.get('department')}")
            print()

    elif args.mode == "rerank":
        if not args.query:
            print("❌ --query is required for rerank mode.")
            return
        rerank(store, args.query)

    elif args.mode == "rag":
        if not args.query:
            print("❌ --query is required for rag mode.")
            return
        run_rag(
            args.query,
            config["LLAMA_API_KEY"],
            setasides=setaside_list,
            naics_codes=naics_list,
            k=args.top_k,
        )

    elif args.mode == "enrich":
        import json
        import boto3

        if not args.path and not args.all and not args.date:
            print("❌ --path, --date, or --all is required for enrich mode.")
            return

        s3 = boto3.client(
            "s3",
            endpoint_url=config.get("MINIO_ENDPOINT", "http://localhost:9000"),
            aws_access_key_id=config.get("MINIO_ACCESS_KEY"),
            aws_secret_access_key=config.get("MINIO_SECRET_KEY"),
            region_name="us-east-1",
        )

        api_key = config.get("SAM_API_KEY")

        def process_record(bucket: str, key: str) -> None:
            obj = s3.get_object(Bucket=bucket, Key=key)
            record = json.loads(obj["Body"].read())
            enriched = enrich_record_with_details(record, s3, bucket, api_key=api_key)
            print(json.dumps({
                "noticeId": record.get("noticeId"),
                "description_key": enriched.get("description_data_key"),
                "attachment_keys": enriched.get("attachment_keys", []),
            }, indent=2))

        if args.all:
            bucket = args.bucket
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if not key.endswith(".json"):
                        continue
                    process_record(bucket, key)
        elif args.date:
            bucket = args.bucket
            from utils.solicitation_assets import list_json_keys_for_date

            for key in list_json_keys_for_date(s3, bucket, args.date):
                process_record(bucket, key)
        else:
            bucket, key = parse_s3_path(args.path, args.bucket)
            process_record(bucket, key)

    elif args.mode == "ragsetup":
        run_rag_setup()

    elif args.mode == "csv-load":
        if not args.csv_file:
            print("❌ --csv-file is required for csv-load mode.")
            return
            
        print("🚀 Loading CSV opportunities and embedding in vector store...")
        csv_agent = CSVOpportunityAgent(args.csv_file, store)
        count = csv_agent.load_and_embed_opportunities()
        
        if count > 0:
            print(f"✅ Successfully loaded and embedded {count} opportunities from CSV")
        else:
            print("❌ No opportunities were loaded from the CSV file")

    elif args.mode == "csv-match":
        if not args.csv_file:
            print("❌ --csv-file is required for csv-match mode.")
            return
        if not args.company_profile:
            print("❌ --company-profile is required for csv-match mode.")
            return
            
        print("🎯 Finding opportunities that match your company profile...")
        csv_agent = CSVOpportunityAgent(args.csv_file, store)
        
        # Force search of existing data only - don't reload CSV
        print("🔍 Checking for existing embeddings in vector store...")
        
        if csv_agent.has_existing_data():
            data_count = csv_agent.get_data_count()
            print(f"📊 Found existing data in vector store (approximately {data_count} opportunities)")
            print("🎯 Searching existing embeddings (not reloading CSV)...")
            
            results = csv_agent.search_existing_opportunities(args.company_profile, args.top_k)
            if not results:
                print("⚠️ No opportunities matched your company profile in the existing data.")
                print("💡 Try adjusting your company profile, using different keywords, or run csv-load first.")
                return
        else:
            print("❌ No existing data found in vector store.")
            print("💡 Please run the following command first to load data:")
            print(f"   pipenv run python main.py --mode csv-load --csv-file {args.csv_file}")
            return
        
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
                print(f"Solicitation #: {metadata.get('solicitation_number', 'N/A')}")
                print(f"Location: {metadata.get('location', 'N/A')}")
                print(f"Link: {metadata.get('link', 'N/A')}")
                print(f"Contact: {metadata.get('primary_contact_name', 'N/A')} ({metadata.get('primary_contact_email', 'N/A')})")
                
                print(f"\n🤖 AI EVALUATION:")
                print("-" * 40)
                print(result['evaluation'])
                print("\n" + "=" * 80)
        else:
            print("❌ No matching opportunities found")

    elif args.mode == "aayeaye":
        search_aayeaye_capabilities(store, k=args.top_k)

if __name__ == "__main__":
    main()
