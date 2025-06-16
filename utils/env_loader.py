from dotenv import load_dotenv
import os

def load_env():
    load_dotenv()
    config = {
        "SAM_API_KEY": os.getenv("SAM_API_KEY"),
        "RERANK_PROMPT_TEMPLATE": os.getenv("RERANK_PROMPT_TEMPLATE"),
        "LLAMA_API_KEY": os.getenv("LLAMA_API_KEY"),
        "MINIO_ACCESS_KEY": os.getenv("MINIO_ACCESS_KEY"),
        "MINIO_SECRET_KEY": os.getenv("MINIO_SECRET_KEY"),
    }
    return config
