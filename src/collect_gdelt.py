"""Collect public news from the GDELT 2.1 DOC API.

GDELT does not require an API key. This module keeps ingestion deliberately
simple: query public articles for issue/geography combinations, normalize the
result into the app schema, deduplicate, and save a local CSV cache.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

ISSUE_QUERIES: dict[str, str] = {
    "affordability / cost of living": '"affordability" OR inflation OR "cost of living" OR groceries OR "utility bills"',
    "housing / rent": 'housing OR rent OR landlord OR eviction OR "property tax"',
    "immigration / public safety": 'immigration OR migrant OR asylum OR border OR "public safety"',
    "AI / tech jobs": 'AI OR "artificial intelligence" OR "tech jobs" OR automation',
    "corruption / competence / trust": 'corruption OR competence OR trust OR crime OR "government failure"',
}

LIVE_QUERY_SEEDS: dict[str, str] = {
    "affordability / cost of living": "inflation",
    "housing / rent": "housing",
    "immigration / public safety": "immigration",
    "AI / tech jobs": "artificial intelligence",
    "corruption / competence / trust": "corruption",
}

NY_GEOGRAPHY_TERMS = [
    "New York",
    "Long Island",
    "Nassau",
    "Suffolk",
    "Westchester",
    "Rockland",
    "Hudson Valley",
    "Queens",
    "Brooklyn",
    "Bronx",
    "Staten Island",
    "Manhattan",
    "Syracuse",
    "Rochester",
    "Buffalo",
    "Albany",
]


def build_gdelt_doc_api_url(query: str, start_datetime: str, end_datetime: str, max_records: int = 50) -> str:
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "startdatetime": start_datetime,
        "enddatetime": end_datetime,
        "maxrecords": max_records,
        "sort": "DateDesc",
    }
    return GDELT_DOC_API + "?" + urlencode(params)


def build_gdelt_timespan_url(query: str, days_back: int, max_records: int = 50) -> str:
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "timespan": f"{days_back}D",
        "maxrecords": max_records,
        "sort": "DateDesc",
    }
    return GDELT_DOC_API + "?" + urlencode(params)


def _gdelt_datetime(value: datetime) -> str:
    return value.strftime("%Y%m%d%H%M%S")


def _query_window(days_back: int) -> tuple[str, str]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    return _gdelt_datetime(start), _gdelt_datetime(end)


def _ny_query(issue_query: str) -> str:
    # Keep the live GDELT request conservative to avoid 429s on broad boolean
    # searches. The broader NY geography list is still used for normalization
    # and can be expanded in production with slower scheduled collection.
    return f"({issue_query}) \"New York\" sourcelang:eng"


def _live_query(issue_area: str) -> str:
    seed = LIVE_QUERY_SEEDS.get(issue_area, "New York")
    return f"{seed} \"New York\" sourcelang:eng"


def fetch_gdelt_issue(
    issue_area: str,
    issue_query: str,
    days_back: int = 14,
    max_records: int = 75,
    timeout_seconds: int = 30,
) -> list[dict]:
    url = build_gdelt_timespan_url(
        _live_query(issue_area),
        days_back=days_back,
        max_records=max_records,
    )
    payload = _request_json(url, timeout_seconds=timeout_seconds)

    rows: list[dict] = []
    for article in payload.get("articles", []):
        title = article.get("title", "")
        url_value = article.get("url", "")
        domain = article.get("domain", "")
        seendate = article.get("seendate", "")
        snippet = f"GDELT public news article from {domain}; seen {seendate}."
        rows.append(
            {
                "date": _normalize_date(seendate),
                "source_name": domain or article.get("sourcecountry", ""),
                "headline": title,
                "snippet": snippet,
                "url": url_value,
                "issue_area": issue_area,
                "geography_refs": _matched_geographies(f"{title} {url_value}"),
                "source_type": "public news via GDELT",
                "domain": domain,
                "seendate": seendate,
                "language": article.get("language", ""),
                "gdelt_query": issue_query,
            }
        )
    return rows


def _request_json(url: str, timeout_seconds: int = 30, attempts: int = 3) -> dict:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": "social-listening-prototype/1.0"})
            with urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8")
            return json.loads(raw)
        except HTTPError as exc:
            last_error = exc
            if exc.code == 429 and attempt < attempts - 1:
                time.sleep(8 * (attempt + 1))
                continue
            raise
        except (URLError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(4 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"GDELT request failed: {last_error}")


def _normalize_date(seendate: str) -> str:
    if not seendate:
        return datetime.now(timezone.utc).date().isoformat()
    compact = seendate.replace("T", "").replace("Z", "").replace(":", "").replace("-", "")
    try:
        return datetime.strptime(compact[:14], "%Y%m%d%H%M%S").date().isoformat()
    except ValueError:
        return datetime.now(timezone.utc).date().isoformat()


def _matched_geographies(text: str) -> str:
    lowered = text.lower()
    matches = [term for term in NY_GEOGRAPHY_TERMS if term.lower() in lowered]
    return "; ".join(matches) if matches else "New York"


def fetch_latest_gdelt_articles(
    days_back: int = 14,
    max_records_per_issue: int = 75,
    output_path: str | Path = "data/gdelt_articles.csv",
) -> pd.DataFrame:
    all_rows: list[dict] = []
    failures: list[str] = []

    for issue_area, query in ISSUE_QUERIES.items():
        try:
            all_rows.extend(
                fetch_gdelt_issue(
                    issue_area,
                    query,
                    days_back=days_back,
                    max_records=max_records_per_issue,
                )
            )
            time.sleep(2)
        except Exception as exc:  # GDELT availability should not break the app.
            failures.append(f"{issue_area}: {exc}")
            time.sleep(2)

    if not all_rows:
        raise RuntimeError("No GDELT articles returned. " + " | ".join(failures))

    df = pd.DataFrame(all_rows)
    df = (
        df.dropna(subset=["headline", "url"])
        .drop_duplicates(subset=["url"])
        .drop_duplicates(subset=["headline"])
        .sort_values("date", ascending=False)
        .reset_index(drop=True)
    )
    df["collection_failures"] = " | ".join(failures)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


def summarize_gdelt_file(path: str | Path = "data/gdelt_articles.csv") -> dict:
    df = pd.read_csv(path)
    failures = ""
    if not df.empty and "collection_failures" in df:
        raw_failures = df["collection_failures"].dropna()
        failures = "" if raw_failures.empty else str(raw_failures.iloc[0])
    return {
        "articles": len(df),
        "start_date": str(pd.to_datetime(df["date"]).min().date()) if not df.empty else None,
        "end_date": str(pd.to_datetime(df["date"]).max().date()) if not df.empty else None,
        "top_sources": df["domain"].fillna(df["source_name"]).value_counts().head(5).to_dict(),
        "failures": failures,
    }


if __name__ == "__main__":
    result = fetch_latest_gdelt_articles()
    print(summarize_gdelt_file())
