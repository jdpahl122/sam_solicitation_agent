import requests
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

        return response.text

    def analyze_solicitation(self, description_text):
        prompt = PromptTemplate.from_template("""
            You are a federal contracting assistant helping a Service-Disabled Veteran-Owned Small Business (SDVOSB) specializing in AI and software services.

            Here is a federal solicitation description:

            {description_text}

            Please do the following:
            1. Summarize what this solicitation is about.
            2. Identify whether it aligns with SDVOSB set-asides or NAICS relevant to technology/AI/software.
            3. Outline what the agency is likely looking for in a qualified vendor.
            4. Suggest 3-5 ways an SDVOSB tech company could position itself to win this opportunity.

            Be thorough but concise.
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
        print("\n✅ Solicitation Analysis Result:")
        print(result)
    except Exception as e:
        print(f"❌ Error: {e}")
