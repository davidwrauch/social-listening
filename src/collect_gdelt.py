"""Optional public-news collection scaffold for future GDELT use.

The Streamlit demo does not call this module. It is included to show where
public-data collection would live in a production-oriented prototype.
"""

from __future__ import annotations

from urllib.parse import urlencode


def build_gdelt_doc_api_url(query: str, start_datetime: str, end_datetime: str, max_records: int = 50) -> str:
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "startdatetime": start_datetime,
        "enddatetime": end_datetime,
        "maxrecords": max_records,
        "sort": "HybridRel",
    }
    return "https://api.gdeltproject.org/api/v2/doc/doc?" + urlencode(params)


def explain_future_collection() -> str:
    return (
        "Future versions could collect public news through GDELT Doc API queries, "
        "normalize articles into the same sample_articles.csv schema, and keep the "
        "classification/scoring rules transparent for auditability."
    )
