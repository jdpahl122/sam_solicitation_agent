from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Dict
import re

ACTIONABLE_TYPES = [
    "Presolicitation",
    "Sources Sought",
    "Combined Synopsis/Solicitation",
    "Solicitation",
]

EXCLUDED_TYPES = [
    "Award Notice",
    "Justification",
    "Special Notice",
    "Sale of Surplus Property",
]


def _parse_date(date_str: str) -> datetime | None:
    """Parse a date string into a timezone-aware UTC datetime."""
    if not date_str:
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(date_str, fmt)
            if not dt.tzinfo:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            continue
    return None


def filter_valid_opportunities(data: List[Dict]) -> List[Dict]:
    """Return only opportunities that meet actionable criteria."""
    now = datetime.now(timezone.utc)
    valid = []

    for opp in data:
        notice_type = opp.get("noticeType") or opp.get("notice_type")
        if not notice_type:
            continue
        if notice_type in EXCLUDED_TYPES or notice_type not in ACTIONABLE_TYPES:
            continue

        deadline_str = (
            opp.get("responseDeadLine")
            or opp.get("responseDeadline")
            or opp.get("response_deadline")
        )
        deadline = _parse_date(deadline_str)
        if not deadline or deadline <= now:
            continue

        valid.append(opp)

    return valid


def summarize_description(text: str) -> str:
    """Return the first two sentences of the text as a summary."""
    if not text:
        return ""
    cleaned = " ".join(text.strip().split())
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return " ".join(sentences[:2]).strip()


def rag_query(user_query: str, store, k: int = 5) -> List[Dict]:
    """Perform a filtered semantic search and return structured summaries."""
    docs = store.index.similarity_search(user_query, k=k * 4)
    results = []
    for doc in docs:
        meta = doc.metadata
        notice_type = meta.get("notice_type")
        response_deadline = meta.get("response_deadline")
        if not filter_valid_opportunities([
            {
                "notice_type": notice_type,
                "response_deadline": response_deadline,
                "noticeType": notice_type,
                "responseDeadLine": response_deadline,
            }
        ]):
            continue

        summary = summarize_description(doc.page_content)
        results.append(
            {
                "summary": summary,
                "title": meta.get("title"),
                "solicitation_number": meta.get("solicitation_number"),
                "naics": meta.get("naics"),
                "response_deadline": response_deadline,
                "link": meta.get("link"),
            }
        )
        if len(results) >= k:
            break
    return results
