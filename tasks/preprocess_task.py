from .base_task import BaseTask

class PreprocessTask(BaseTask):
    def execute(self, opportunities):
        processed_docs = []
        for opp in opportunities:
            doc = f"""
Title: {opp.get('title')}
Notice ID: {opp.get('noticeId')}
Description: {opp.get('description')}
NAICS: {opp.get('naicsCodes')}
Set-Aside: {opp.get('typeOfSetAsideDescription')}
"""
            processed_docs.append(doc)
        return processed_docs
