from .base_task import BaseTask
from utils.rag_helpers import filter_valid_opportunities

class PreprocessTask(BaseTask):
    def execute(self, opportunities):
        # Filter out invalid/expired opportunities before embedding
        opportunities = filter_valid_opportunities(opportunities)
        processed_docs = []

        for opp in opportunities:
            # Skip inactive or archived opportunities when possible
            archive_date = opp.get("archiveDate")
            if archive_date:
                try:
                    from datetime import datetime

                    ad = datetime.strptime(archive_date, "%m/%d/%Y")
                    if ad < datetime.now():
                        continue
                except Exception:
                    pass

            if opp.get("active") is False:
                continue

            description = opp.get("description") or ""
            attachments_text = opp.get("attachmentText") or ""
            title = opp.get("title") or "Untitled"
            solicitation_number = opp.get("solicitationNumber") or "Unknown"
            link = opp.get("uiLink") or ""
            naics = opp.get("naicsCode") or "Unknown"
            setaside = opp.get("typeOfSetAsideDescription") or "None"
            notice_id = opp.get("noticeId") or "Unknown"

            text = f"{description}\n\n{attachments_text}".strip()

            metadata = {
                "title": title,
                "solicitation_number": solicitation_number,
                "link": link,
                "naics": naics,
                "setaside": setaside,
                "posted_date": opp.get("postedDate") or "Unknown",
                "notice_id": notice_id,
                "notice_type": opp.get("noticeType") or "",
                "response_deadline": opp.get("responseDeadLine") or "",
            }

            processed_docs.append({
                "text": text,
                "metadata": metadata
            })

        return processed_docs
