from .base_task import BaseTask

class PreprocessTask(BaseTask):
    def execute(self, opportunities):
        processed_docs = []

        for opp in opportunities:
            description = opp.get("description") or ""
            title = opp.get("title", "Untitled")
            solicitation_number = opp.get("solicitationNumber", "Unknown")
            link = opp.get("uiLink", "")
            naics = opp.get("naicsCode", "Unknown")
            setaside = opp.get("typeOfSetAsideDescription", "None")

            text = f"{title}\n\nSolicitation Number: {solicitation_number}\nNAICS: {naics}\nSet-Aside: {setaside}\n\n{description}"

            metadata = {
                "title": title,
                "solicitation_number": solicitation_number,
                "link": link,
                "naics": naics,
                "setaside": setaside,
                "posted_date": opp.get("postedDate", "Unknown"),
            }

            processed_docs.append({
                "text": text,
                "metadata": metadata
            })

        return processed_docs
