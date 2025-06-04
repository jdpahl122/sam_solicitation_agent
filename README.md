# 🛡️ SAM Solicitation Agent

**An intelligent assistant that pulls, processes, stores, and finds the best federal contracting opportunities for SDVOSB and technology companies.**  
Built using LangChain, Ollama, and FAISS vector storage.

## Features

- Pulls live solicitations from [SAM.gov](https://sam.gov)
- Preprocesses and embeds opportunities using local LLMs
- Stores embeddings with full metadata in a persistent FAISS vector database
- Semantic search across all opportunities
- Intelligent reranking of results based on your company qualifications
- Modular architecture: ingestion chain, search chain, rerank chain
- CLI interface for flexible operation (`ingest`, `search`, `rerank`)
- Retrieval-Augmented Generation (RAG) mode for conversational answers
- Set-aside filtering and top-N result limiting
- Parallel ingestion for faster pulls from SAM.gov
- Automatically initializes the FAISS index if none exists
- Posted dates displayed in search and RAG results


## Requirements

- Python 3.10+
- `pipenv`
- [Ollama](https://ollama.ai/) installed locally
- FAISS via `langchain_community`
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
```

The agent requires your **SAM.gov API key**. The `LLAMA_API_KEY` is only needed
for the optional RAG mode and solicitation overview script.

## Initializing the FAISS Store

Instantiate `FaissStore` to create or load the vector index in `./vector_store`.
If the files are missing or corrupted they will be recreated automatically:

```python
from rag.faiss_store import FaissStore

# creates ./vector_store/index.faiss on first run
store = FaissStore()
```

Then run the ingest mode to populate it:

```bash
python main.py --mode ingest
```

The index will persist for future searches and RAG responses. If you encounter
errors loading the store, delete the `vector_store` directory and rerun the
ingest mode.

## Usage

The CLI is modular — you can **ingest**, **search**, **rerank** or use a simple
**RAG** mode independently:

### 1. Ingest Opportunities

```bash
python main.py --mode ingest
```

### 2. Semantic Search

```bash
python main.py --mode search --query "AI contracting work for a small business"
```

### 3. Rerank with LLM Intelligence

```bash
python main.py --mode rerank --query "AI contracting work for a small business"
```

### 4. RAG Mode

Use a lightweight Retrieval-Augmented Generation mode. Requires `LLAMA_API_KEY`.

```bash
python main.py --mode rag --query "Explain AI contract opportunities in cyber"
```

### 5. Solicitation Overview

Summarize a single solicitation by its notice ID:

```bash
python solicitation_overview.py <notice_id>
```


## Architecture

| Component | Purpose |
|:--|:--|
| `agents/solicitation_agent.py` | Pull → preprocess → embed |
| `chains/semantic_search_chain.py` | Search FAISS vectorstore |
| `chains/rerank_chain.py` | Rerank results with LLM |
| `rag/faiss_store.py` | Persistent FAISS vector DB |
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
python main.py --mode ingest
python main.py --mode search --query "Cybersecurity support for government agencies"
python main.py --mode rerank --query "Cybersecurity support for government agencies"
python solicitation_overview.py 02aa3325308f491d959ba968898accd6
```


# About

Built by [Your Name] to supercharge federal contracting discovery for SDVOSBs and small businesses in tech.

> "Find the contracts that actually fit you — without digging through thousands of opportunities."


# Links

- [SAM.gov API Docs](https://open.gsa.gov/api/sam/opportunities-api/)
- [Ollama](https://ollama.ai/)
- [LangChain](https://python.langchain.com/)

