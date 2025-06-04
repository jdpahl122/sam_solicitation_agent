from langchain.schema import Document
from utils.rag_helpers import filter_valid_opportunities, summarize_description, rag_query

class DummyIndex:
    def __init__(self, docs):
        self.docs = docs

    def similarity_search(self, query, k=10):
        return self.docs

class DummyStore:
    def __init__(self, docs):
        self.index = DummyIndex(docs)

def test_filter_valid_opportunities():
    data = [
        {"noticeType": "Solicitation", "responseDeadLine": "12/31/2999"},
        {"noticeType": "Award Notice", "responseDeadLine": "12/31/2999"},
        {"noticeType": "Solicitation", "responseDeadLine": "01/01/2000"},
        {"noticeType": "Solicitation"},
    ]
    result = filter_valid_opportunities(data)
    assert len(result) == 1
    assert result[0]["noticeType"] == "Solicitation"


def test_summarize_description():
    text = "First sentence. Second sentence. Third sentence."
    assert summarize_description(text) == "First sentence. Second sentence."


def test_rag_query():
    docs = [
        Document(page_content="Description one. More text.", metadata={
            "title": "A",
            "solicitation_number": "1",
            "naics": "541511",
            "link": "http://example.com",
            "notice_type": "Solicitation",
            "response_deadline": "12/31/2999",
        }),
        Document(page_content="Bad notice", metadata={
            "title": "B",
            "solicitation_number": "2",
            "naics": "541512",
            "link": "http://example.com/2",
            "notice_type": "Award Notice",
            "response_deadline": "12/31/2999",
        }),
    ]
    store = DummyStore(docs)
    results = rag_query("test", store, k=5)
    assert len(results) == 1
    assert results[0]["title"] == "A"
