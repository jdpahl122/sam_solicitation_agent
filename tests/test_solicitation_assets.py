import json
from utils.solicitation_assets import (
    enrich_record_with_details,
    parse_s3_path,
    date_to_prefix,
    list_json_keys_for_date,
)


class DummyS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body):
        self.objects[(Bucket, Key)] = Body


def test_enrich_record_with_details(monkeypatch):
    record = {
        "noticeId": "abc123",
        "postedDate": "2025-06-16",
        "description": "http://example.com/desc",
        "resourceLinks": [
            "http://example.com/file1/download",
            "http://example.com/file2/download",
        ],
    }

    class DummyResp:
        def __init__(self, content, json_data=None):
            self.content = content
            self._json = json_data
            self.headers = {}

        def raise_for_status(self):
            pass

        def json(self):
            if self._json is None:
                raise ValueError()
            return self._json

    calls = []

    def mock_get(url):
        calls.append(url)
        if url == "http://example.com/desc":
            return DummyResp(b"{}", json_data={"description": "full text"})
        return DummyResp(b"data")

    monkeypatch.setattr("requests.get", mock_get)

    s3 = DummyS3()
    enriched = enrich_record_with_details(record, s3, "bucket", dry_run=False)

    assert enriched["description_data_key"].endswith("description.json")
    assert len(enriched["attachment_keys"]) == 2
    assert len(s3.objects) == 3
    assert calls[0] == "http://example.com/desc"


def test_parse_s3_path():
    b, k = parse_s3_path("sam-archive/2025/06/16/foo.json")
    assert b == "sam-archive"
    assert k == "2025/06/16/foo.json"

    b, k = parse_s3_path("2025/06/16/foo.json", "sam-archive")
    assert b == "sam-archive"
    assert k == "2025/06/16/foo.json"


def test_date_to_prefix():
    assert date_to_prefix("2025-06-17") == "2025/06/17/"


class DummyListS3:
    def __init__(self, pages):
        self.pages = pages

    class Pager:
        def __init__(self, pages):
            self.pages = pages

        def paginate(self, **kwargs):
            return iter(self.pages)

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return DummyListS3.Pager(self.pages)


def test_list_json_keys_for_date():
    pages = [
        {"Contents": [
            {"Key": "2025/06/17/a.json"},
            {"Key": "2025/06/17/b.txt"},
        ]},
        {"Contents": [
            {"Key": "2025/06/17/c.json"}
        ]},
    ]
    s3 = DummyListS3(pages)
    keys = list(list_json_keys_for_date(s3, "bucket", "2025-06-17"))
    assert keys == ["2025/06/17/a.json", "2025/06/17/c.json"]
