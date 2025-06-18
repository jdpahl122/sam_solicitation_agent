import argparse
from utils.env_loader import load_env
from utils.solicitation_assets import enrich_record_with_details, parse_s3_path
from agents.solicitation_agent import SolicitationAgent
from chains.semantic_search_chain import SemanticSearchChain
from chains.rerank_chain import RerankChain
from rag.faiss_store import FaissStore
from llm import LlamaRAG
from scripts.rag_setup import run as run_rag_setup

def ingest(config, store):
    agent = SolicitationAgent(config, store)
    agent.run()

def search(store, query, k=20, setasides=None, naics_codes=None):
    search_chain = SemanticSearchChain(store.index)
    results = search_chain.execute(query, k=k)
    if setasides:
        allowed = {sa.lower() for sa in setasides}
        results = [d for d in results if d.metadata.get("setaside", "").lower() in allowed]
    if naics_codes:
        allowed_naics = {code.strip() for code in naics_codes}
        results = [d for d in results if d.metadata.get("naics") in allowed_naics]
    return results[:k]

def rerank(store, query):
    search_chain = SemanticSearchChain(store.index)
    results = search_chain.execute(query)

    rerank_chain = RerankChain()
    top_5 = rerank_chain.execute(query, results)
    print("\n✅ Top Recommended Opportunities:\n")
    print(top_5)

def run_rag(query, api_key, setasides=None, naics_codes=None, k=20):
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

def main():
    parser = argparse.ArgumentParser(description="SAM Solicitation Agent CLI")
    parser.add_argument(
        "--mode",
        choices=["ingest", "search", "rerank", "rag", "enrich", "ragsetup"],
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

    args = parser.parse_args()
    setaside_list = None
    if args.setaside:
        setaside_list = [s.strip() for s in args.setaside.split(',') if s.strip()]
    naics_list = None
    if args.naics:
        naics_list = [c.strip() for c in args.naics.split(',') if c.strip()]

    config = load_env()
    store = FaissStore()

    if args.mode == "ingest":
        ingest(config, store)

    elif args.mode == "search":
        if not args.query:
            print("❌ --query is required for search mode.")
            return
        results = search(store, args.query, k=20, setasides=setaside_list, naics_codes=naics_list)
        print(f"\n✅ Top {len(results)} Search Results:\n")
        for i, doc in enumerate(results, 1):
            meta = doc.metadata
            print(f"--- [{i}] ---")
            print(f"Title: {meta.get('title')}")
            print(f"Solicitation #: {meta.get('solicitation_number')}")
            print(f"Posted: {meta.get('posted_date')}")
            print(f"Link: {meta.get('link')}")
            print(f"NAICS: {meta.get('naics')}")
            print(f"Set-Aside: {meta.get('setaside')}")
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
            k=20,
        )

    elif args.mode == "enrich":
        import json
        import boto3

        if not args.path and not args.all and not args.date:
            print("❌ --path, --date, or --all is required for enrich mode.")
            return

        s3 = boto3.client(
            "s3",
            endpoint_url="http://localhost:9000",
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

if __name__ == "__main__":
    main()