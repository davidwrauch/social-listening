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

NY_GEOGRAPHY_TERMS = [
    "New York",
    "NYC",
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

ISSUE_KEYWORD_TERMS: dict[str, list[str]] = {
    "affordability / cost of living": [
        "affordability",
        "inflation",
        "cost of living",
        "groceries",
        "utility bills",
    ],
    "housing / rent": ["housing", "rent", "landlord", "eviction", "property tax"],
    "immigration / public safety": [
        "immigration",
        "migrant",
        "asylum",
        "border",
        "public safety",
    ],
    "AI / tech jobs": ["AI", "artificial intelligence", "tech jobs", "automation"],
    "corruption / competence / trust": [
        "corruption",
        "competence",
        "trust",
        "crime",
        "government failure",
    ],
}

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
    geography_query = " OR ".join(_quote_if_needed(term) for term in NY_GEOGRAPHY_TERMS)
    return f"({issue_query}) AND ({geography_query}) sourcelang:eng"


def _ny_term_query(issue_query: str, geography_term: str) -> str:
    return f"({issue_query}) AND {_quote_if_needed(geography_term)} sourcelang:eng"


def _quote_if_needed(term: str) -> str:
    return f'"{term}"' if " " in term else term


def fetch_gdelt_issue(
    issue_area: str,
    issue_query: str,
    days_back: int = 14,
    max_records: int = 75,
    timeout_seconds: int = 30,
) -> list[dict]:
    rows: list[dict] = []
    query = _ny_query(issue_query)
    url = build_gdelt_timespan_url(query, days_back=days_back, max_records=max_records)
    response_data = _request_json(url, timeout_seconds=timeout_seconds, attempts=2)
    for article in response_data.get("articles", []):
        title = article.get("title", "")
        url_value = article.get("url", "")
        domain = article.get("domain", "")
        seendate = article.get("seendate", "")
        snippet = _build_snippet(article, "New York geography block", issue_query)
        geography_matches = _matched_terms(f"{title} {snippet}", NY_GEOGRAPHY_TERMS)
        issue_matches = _matched_terms(f"{title} {snippet}", ISSUE_KEYWORD_TERMS[issue_area])
        relevance_score = _relevance_score(
            title=title,
            snippet=snippet,
            issue_terms=ISSUE_KEYWORD_TERMS[issue_area],
        )
        rows.append(
            {
                "date": _normalize_date(seendate),
                "source_name": domain or article.get("sourcecountry", ""),
                "headline": title,
                "snippet": snippet,
                "url": url_value,
                "issue_area": issue_area,
                "geography_refs": "; ".join(geography_matches),
                "geography_matches": "; ".join(geography_matches),
                "issue_keyword_matches": "; ".join(issue_matches),
                "relevance_score": relevance_score,
                "source_type": "public news via GDELT",
                "domain": domain,
                "seendate": seendate,
                "language": article.get("language", ""),
                "gdelt_query": query,
            }
        )
    return rows


def _build_snippet(article: dict, query_geography: str = "", issue_query: str = "") -> str:
    parts = [
        article.get("title", ""),
        f"GDELT query geography: {query_geography}" if query_geography else "",
        f"GDELT query issue terms: {issue_query}" if issue_query else "",
        article.get("domain", ""),
        article.get("sourcecountry", ""),
        article.get("seendate", ""),
        article.get("url", ""),
    ]
    return " | ".join(str(part) for part in parts if part)


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
                time.sleep(5 * (attempt + 1))
                continue
            raise
        except (URLError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(3 * (attempt + 1))
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
    return "; ".join(_matched_terms(text, NY_GEOGRAPHY_TERMS))


def _matched_terms(text: str, terms: list[str]) -> list[str]:
    lowered = str(text).lower()
    return [term for term in terms if term.lower() in lowered]


def _relevance_score(title: str, snippet: str, issue_terms: list[str]) -> int:
    title_geo = bool(_matched_terms(title, NY_GEOGRAPHY_TERMS))
    snippet_geo = bool(_matched_terms(snippet, NY_GEOGRAPHY_TERMS))
    title_issue = bool(_matched_terms(title, issue_terms))
    snippet_issue = bool(_matched_terms(snippet, issue_terms))
    return int(title_geo) * 2 + int(snippet_geo) + int(title_issue) + int(snippet_issue)


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
            time.sleep(1)
        except Exception as exc:  # GDELT availability should not break the app.
            failures.append(f"{issue_area}: {exc}")
            time.sleep(1)

    if not all_rows:
        raise RuntimeError("No GDELT articles returned. " + " | ".join(failures))

    df = pd.DataFrame(all_rows)
    rows_before_filtering = len(df)
    df = df.dropna(subset=["headline", "url"]).drop_duplicates(subset=["url"]).drop_duplicates(subset=["headline"])
    rows_after_dedupe = len(df)
    df = (
        df.query("language == 'English'")
        .query("relevance_score >= 2")
        .sort_values("date", ascending=False)
        .reset_index(drop=True)
    )
    df["collection_failures"] = " | ".join(failures)
    df["rows_before_filtering"] = rows_before_filtering
    df["rows_after_dedupe"] = rows_after_dedupe
    df["rows_after_filtering"] = len(df)

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
        "rows_before_filtering": int(df["rows_before_filtering"].iloc[0])
        if "rows_before_filtering" in df and not df.empty
        else len(df),
        "rows_after_dedupe": int(df["rows_after_dedupe"].iloc[0])
        if "rows_after_dedupe" in df and not df.empty
        else len(df),
        "rows_after_filtering": int(df["rows_after_filtering"].iloc[0])
        if "rows_after_filtering" in df and not df.empty
        else len(df),
        "start_date": str(pd.to_datetime(df["date"]).min().date()) if not df.empty else None,
        "end_date": str(pd.to_datetime(df["date"]).max().date()) if not df.empty else None,
        "top_geography_matches": _top_semicolon_values(df, "geography_matches"),
        "top_sources": df["domain"].fillna(df["source_name"]).value_counts().head(5).to_dict(),
        "failures": failures,
    }


def _top_semicolon_values(df: pd.DataFrame, column: str, limit: int = 5) -> dict[str, int]:
    if column not in df:
        return {}
    values: list[str] = []
    for raw_value in df[column].dropna():
        values.extend(value.strip() for value in str(raw_value).split(";") if value.strip())
    return {str(key): int(value) for key, value in pd.Series(values).value_counts().head(limit).items()}


if __name__ == "__main__":
    result = fetch_latest_gdelt_articles()
    print(summarize_gdelt_file())
