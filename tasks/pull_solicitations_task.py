from utils.sam_api import SamAPIClient
from .base_task import BaseTask
from datetime import datetime, timedelta

class PullSolicitationsTask(BaseTask):
    def __init__(self, api_key):
        self.client = SamAPIClient(api_key)

    def execute(self):
        opportunities = self.client.search_opportunities(limit=100)

        return opportunities
