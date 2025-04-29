from dotenv import load_dotenv
import os

def load_env():
    load_dotenv()
    config = {
        "SAM_API_KEY": os.getenv("SAM_API_KEY"),
    }
    return config
