# 🛡️ SAM Solicitation Agent

**An intelligent assistant that pulls, processes, stores, and finds the best federal contracting opportunities for SDVOSB and technology companies.**  
Built using LangChain, Ollama, and a Milvus vector database.

## Features

- Pulls live solicitations from [SAM.gov](https://sam.gov)
- **🆕 CSV Opportunity Matching**: Load ContractOpportunitiesFullCSV.csv and use AI to find the best matches for your company profile
- Preprocesses and embeds opportunities using local LLMs
- Stores embeddings with full metadata in a persistent Milvus vector database
- Semantic search across all opportunities
- Intelligent reranking of results based on your company qualifications
- **🆕 AI-Powered Opportunity Evaluation**: Uses LLM to score opportunities based on set-aside match, capabilities, NAICS codes, competition, and more
- Modular architecture: ingestion chain, search chain, rerank chain
- CLI interface for flexible operation (`ingest`, `search`, `rerank`, `csv-load`, `csv-match`)
- Retrieval-Augmented Generation (RAG) mode for conversational answers
- Set-aside and NAICS code filtering with top-N result limiting
- Parallel ingestion for faster pulls from SAM.gov
- Automatically initializes the Milvus collection if none exists
- Posted dates displayed in search and RAG results
- Cleans old vector data so only active solicitations remain
- Archives raw solicitation JSON to a local MinIO object store


## Requirements

- Python 3.10+
- `pipenv`
- [Ollama](https://ollama.ai/) installed locally
- Milvus via `langchain_community`
- SAM.gov API Key


## Installation

```bash
git clone https://github.com/yourrepo/sam_solicitation_agent.git
cd sam_solicitation_agent
pipenv install
pipenv shell
```

Create a `.env` file:

```dotenv
SAM_API_KEY=your-sam-api-key-here
# Optional: API key for the Llama RAG mode
LLAMA_API_KEY=your-ollama-key-here
MINIO_ACCESS_KEY=minio-access
MINIO_SECRET_KEY=minio-secret
MINIO_ENDPOINT=http://localhost:9000
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

The agent requires your **SAM.gov API key**. The `LLAMA_API_KEY` is only needed
for the optional RAG mode and solicitation overview script.

## Initializing the Milvus Store

Instantiate `MilvusStore` to connect to the Milvus collection.
If the files are missing or corrupted they will be recreated automatically:

```python
from rag.milvus_store import MilvusStore

# connects to Milvus on first run
store = MilvusStore()
```

Then run the ingest mode to populate it. Each run now clears any existing
documents so only active solicitations remain:

```bash
pipenv run python main.py --mode ingest
```

The collection will persist for future searches and RAG responses. If you
encounter connection errors, ensure the Milvus service is running and reachable
before rerunning the ingest mode.

## Usage

The CLI is modular — you can **ingest**, **search**, **rerank** or use a simple
**RAG** mode independently:

### 1. Ingest Opportunities (clears old entries)

```bash
pipenv run python main.py --mode ingest
```

### 2. Semantic Search

```bash
pipenv run python main.py --mode search --query "AI contracting work for a small business"
# Filter by set-aside or NAICS codes
pipenv run python main.py --mode search --query "AI contracting" --setaside "8(a) Set-Aside" --naics "541511,541512"
```

### 3. Rerank with LLM Intelligence

```bash
pipenv run python main.py --mode rerank --query "AI contracting work for a small business"
```

### 4. RAG Mode

Use a lightweight Retrieval-Augmented Generation mode. Requires `LLAMA_API_KEY`.

```bash
pipenv run python main.py --mode rag --query "Explain AI contract opportunities in cyber"
```

### 5. Enrich Stored Records

Download long descriptions and all attachment files for a saved JSON record in MinIO.

```bash
pipenv run python main.py --mode enrich --path sam-archive/2025/06/17/<notice_id>.json
```

Process the entire bucket at once:

```bash
pipenv run python main.py --mode enrich --all
```

Process only a single day's records:

```bash
pipenv run python main.py --mode enrich --date 2025-06-17
```

### 6. CSV Opportunity Matching (NEW!)

Load opportunities from ContractOpportunitiesFullCSV.csv and find the best matches for your company using AI evaluation:

#### Load CSV Data and Embed in Vector Store

```bash
pipenv run python main.py --mode csv-load --csv-file /path/to/ContractOpportunitiesFullCSV.csv
```

#### Find Matching Opportunities with AI Evaluation

```bash
pipenv run python main.py --mode csv-match \
  --csv-file /path/to/ContractOpportunitiesFullCSV.csv \
  --company-profile "I'm a Service Disabled Veteran owned small business with a focus in AI, ML, Software Architecture, and VAR opportunities related to software products. I'm mostly looking for set-asides that match my skillset" \
  --top-k 5
```

This will:
1. Load and embed all active opportunities from the CSV
2. Use an LLM to evaluate each opportunity against your company profile
3. Score opportunities based on set-aside match, capability alignment, NAICS relevance, competition level, contract value, and geographic fit
4. Return the top-ranked opportunities with detailed AI analysis and actionable recommendations

### 7. Solicitation Overview

Summarize a single solicitation by its notice ID:

```bash
pipenv run python solicitation_overview.py <notice_id>
```


## Architecture

| Component | Purpose |
|:--|:--|
| `agents/solicitation_agent.py` | Pull → preprocess → embed |
| `chains/semantic_search_chain.py` | Search Milvus vectorstore |
| `chains/rerank_chain.py` | Rerank results with LLM |
| `rag/milvus_store.py` | Persistent Milvus vector DB |
| `tasks/` | Modular task units (pull, preprocess, search) |
| `prompts/` | Rerank prompt templates |
| `tests/` | Unit tests |

## Testing

Run the small test suite with:

```bash
pipenv run pytest -q
```


## Future Upgrades

- Add filtering (e.g., only SDVOSB or specific NAICS)
- Export matched solicitations to CSV/JSON
- Support automatic nightly ingestion (CRON jobs)
- Implement UI dashboard (streamlit / fastapi)


## Example

```bash
# Traditional SAM.gov API workflow
pipenv run python main.py --mode ingest
pipenv run python main.py --mode search --query "Cybersecurity support for government agencies"
pipenv run python main.py --mode rerank --query "Cybersecurity support for government agencies"
pipenv run python main.py --mode rag --query "Cybersecurity" --naics "541519" --setaside "Total Small Business Set-Aside (FAR 19.5)"

# NEW: CSV opportunity matching workflow
pipenv run python main.py --mode csv-load --csv-file ./ContractOpportunitiesFullCSV.csv
pipenv run python main.py --mode csv-match \
  --csv-file ./ContractOpportunitiesFullCSV.csv \
  --company-profile "Service Disabled Veteran owned small business specializing in AI/ML software development and cloud solutions" \
  --top-k 10

# Solicitation details
pipenv run python solicitation_overview.py 02aa3325308f491d959ba968898accd6
```


# About

Built by Jonathan Pahl to supercharge federal contracting discovery for SDVOSBs and small businesses in tech.

> "Find the contracts that actually fit you — without digging through thousands of opportunities."


# Links

- [SAM.gov API Docs](https://open.gsa.gov/api/sam/opportunities-api/)
- [Ollama](https://ollama.ai/)
- [LangChain](https://python.langchain.com/)
