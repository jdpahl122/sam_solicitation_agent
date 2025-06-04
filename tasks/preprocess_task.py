from .base_task import BaseTask

class PreprocessTask(BaseTask):
    def execute(self, opportunities):
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
            title = opp.get("title", "Untitled")
            solicitation_number = opp.get("solicitationNumber", "Unknown")
            link = opp.get("uiLink", "")
            naics = opp.get("naicsCode", "Unknown")
            setaside = opp.get("typeOfSetAsideDescription", "None")
            notice_id = opp.get("noticeId", "Unknown")

            text = (
                f"{title}\n\n"
                f"Solicitation Number: {solicitation_number}\n"
                f"NAICS: {naics}\n"
                f"Set-Aside: {setaside}\n\n"
                f"{description}"
            )

            metadata = {
                "title": title,
                "solicitation_number": solicitation_number,
                "link": link,
                "naics": naics,
                "setaside": setaside,
                "posted_date": opp.get("postedDate", "Unknown"),
                "notice_id": notice_id,
            }

            processed_docs.append({
                "text": text,
                "metadata": metadata
            })

        return processed_docs
