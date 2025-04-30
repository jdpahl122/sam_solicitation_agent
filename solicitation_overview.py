import requests
import json
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.env_loader import load_env


class SolicitationOverview:
    BASE_URL = "https://api.sam.gov/prod/opportunity/v2/opportunities"

    def __init__(self, api_key):
        self.api_key = api_key
        self.llm = Ollama(model="llama3")

    def fetch_by_notice_id(self, notice_id):
        url = f"https://api.sam.gov/prod/opportunities/v1/noticedesc?noticeid={notice_id}&api_key={self.api_key}"
        headers = {"Accept": "application/json"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        try:
            data = response.json()

            # Step 1: Try accessing 'description' directly
            description_raw = data.get("description", "")
            if not description_raw:
                raise ValueError("No description found in API response.")

            # Step 2: If it's a string, try to parse it as JSON
            if isinstance(description_raw, str):
                try:
                    description_obj = json.loads(description_raw)
                    description_text = description_obj.get("description", "")
                except json.JSONDecodeError:
                    # If it's not JSON, just return the string as-is
                    description_text = description_raw
            elif isinstance(description_raw, dict):
                description_text = description_raw.get("description", "")
            else:
                raise TypeError("Unknown format for description field.")

            if not description_text.strip():
                raise ValueError("Description field is empty after parsing.")

            return description_text

        except Exception as e:
            raise RuntimeError(f"Failed to parse description: {e}")

    def analyze_solicitation(self, description_text):
        prompt = PromptTemplate.from_template("""
You are a federal contracting advisor helping a Service-Disabled Veteran-Owned Small Business (SDVOSB) specializing in:
- Artificial Intelligence and Machine Learning
- Secure Infrastructure
- MLOps and model monitoring
- Software and cloud platform development
- Compliance (RMF, NIST 800-53)

Given the following federal solicitation description:

{description_text}

Extract and return a clean, human-readable summary with the following fields:

1. **Opportunity Summary** – A concise 2–3 sentence summary of what this opportunity is about.
2. **Relevance to SDVOSB** – Is it set aside for SDVOSB or aligned with NAICS codes in tech/AI/compliance?
3. **What the agency needs** – Bullet list of vendor capabilities being sought.
4. **How this SDVOSB could win** – 3–5 recommendations specific to a small business with this technical background.
5. **Red Flags** – Note any issues that could make this a poor fit (optional).

Respond in plain markdown for easy display.
        """)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.invoke({"description_text": description_text})


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("❌ Please provide a solicitation notice ID.")
        sys.exit(1)

    notice_id = sys.argv[1]
    config = load_env()
    overviewer = SolicitationOverview(config["SAM_API_KEY"])

    print(f"📄 Fetching full description for Notice ID: {notice_id}")
    try:
        description = overviewer.fetch_by_notice_id(notice_id)
        print("\n🤖 Analyzing with LLM...")
        result = overviewer.analyze_solicitation(description)

        print("\n✅ Solicitation Summary:")
        print(result["text"])  # Will show markdown-style fields

    except Exception as e:
        print(f"❌ Error: {e}")
