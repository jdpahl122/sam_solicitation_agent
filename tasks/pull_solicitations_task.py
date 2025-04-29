from utils.sam_api import SamAPIClient
from .base_task import BaseTask

class PullSolicitationsTask(BaseTask):
    def __init__(self, api_key):
        self.client = SamAPIClient(api_key)

    def execute(self):
        opportunities = self.client.search_opportunities()
        return opportunities
