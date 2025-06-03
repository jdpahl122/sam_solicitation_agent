import requests
from tasks.pull_solicitations_task import PullSolicitationsTask

class MockResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"Status {self.status_code}")

def test_execute_returns_opportunities(monkeypatch):
    call_params = []

    def mock_get(url, params=None, **kwargs):
        call_params.append(params)
        if params.get("limit") == 1:
            return MockResponse({"totalRecords": 2})
        else:
            return MockResponse({"opportunitiesData": [
                {"id": "opp1"},
                {"id": "opp2"}
            ]})

    monkeypatch.setattr(requests, "get", mock_get)
    task = PullSolicitationsTask(api_key="dummy")
    result = task.execute()

    assert result == [{"id": "opp1"}, {"id": "opp2"}]
    # One request for total records and one for the page
    assert len(call_params) == 2
