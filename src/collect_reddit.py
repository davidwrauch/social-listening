"""Collect public Reddit posts for aggregate narrative monitoring.

This collector uses Reddit's public JSON endpoints with a clear User-Agent.
It does not require OAuth, does not collect private data, and saves only
post-level public discussion signals for aggregate research analysis.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import pandas as pd

from src.classify_topics import ISSUE_KEYWORDS
from src.regions import NY_GEOGRAPHY_TERMS, REGION_KEYWORDS, normalize_regions

USER_AGENT = "social-listening-prototype/0.1 by davidwrauch"
REDDIT_BASE = "https://www.reddit.com"

NY_SUBREDDITS = ["nyc", "newyork", "longisland", "buffalo", "Rochester", "Syracuse", "Albany", "AskNYC"]
OPTIONAL_SUBREDDITS = ["politics"]
ISSUE_TERMS = [
    "affordability",
    "inflation",
    "cost of living",
    "rent",
    "housing",
    "landlord",
    "eviction",
    "property tax",
    "immigration",
    "migrant",
    "asylum",
    "public safety",
    "crime",
    "AI",
    "artificial intelligence",
    "tech jobs",
    "automation",
    "corruption",
    "trust",
    "government",
    "competence",
]
NY_SPECIFIC_SUBREDDITS = {sub.lower() for sub in NY_SUBREDDITS}
SUBREDDIT_REGION = {
    "nyc": "NYC",
    "asknyc": "NYC",
    "newyork": "NYC",
    "longisland": "Long Island",
    "buffalo": "Western NY",
    "rochester": "Western NY",
    "syracuse": "Central NY",
    "albany": "Capital Region",
}


def fetch_latest_reddit_posts(
    days_back: int = 14,
    output_path: str | Path = "data/reddit_posts.csv",
    limit_per_request: int = 100,
    include_search: bool = False,
    timeout_seconds: int = 20,
) -> pd.DataFrame:
    rows: list[dict] = []
    failures: list[str] = []

    for subreddit in [*NY_SUBREDDITS, *OPTIONAL_SUBREDDITS]:
        urls = [f"{REDDIT_BASE}/r/{subreddit}/new.json?limit={limit_per_request}"]
        if include_search:
            for query in _search_queries():
                encoded = quote_plus(query)
                urls.append(
                    f"{REDDIT_BASE}/r/{subreddit}/search.json?q={encoded}&restrict_sr=1&sort=new&limit={limit_per_request}"
                )

        for url in urls:
            try:
                response_data = _request_json(url, timeout_seconds=timeout_seconds)
                rows.extend(_normalize_listing(response_data, subreddit=subreddit, days_back=days_back))
            except Exception as exc:  # Reddit availability should not break the app.
                failures.append(f"r/{subreddit}: {exc}")
            time.sleep(0.6)

    if not rows:
        raise RuntimeError("No Reddit posts returned. " + " | ".join(failures[:5]))

    df = pd.DataFrame(rows)
    rows_before_filtering = len(df)
    df = (
        df.dropna(subset=["headline", "url"])
        .drop_duplicates(subset=["url"])
        .drop_duplicates(subset=["headline"])
        .query("relevance_score >= 2")
        .sort_values("date", ascending=False)
        .reset_index(drop=True)
    )
    df["collection_failures"] = " | ".join(failures)
    df["rows_before_filtering"] = rows_before_filtering
    df["rows_after_filtering"] = len(df)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


def _search_queries() -> list[str]:
    grouped = [
        " OR ".join([_quote(term) for term in ISSUE_TERMS[:5]]),
        " OR ".join([_quote(term) for term in ISSUE_TERMS[5:10]]),
        " OR ".join([_quote(term) for term in ISSUE_TERMS[10:15]]),
        " OR ".join([_quote(term) for term in ISSUE_TERMS[15:]]),
    ]
    return grouped


def _normalize_listing(response_data: dict, subreddit: str, days_back: int) -> list[dict]:
    cutoff = datetime.now(timezone.utc).timestamp() - days_back * 86400
    rows: list[dict] = []
    for child in response_data.get("data", {}).get("children", []):
        post = child.get("data", {})
        created = float(post.get("created_utc") or 0)
        if created and created < cutoff:
            continue
        title = str(post.get("title") or "").strip()
        body = str(post.get("selftext") or "").strip()
        permalink = post.get("permalink") or ""
        url = f"{REDDIT_BASE}{permalink}" if permalink else str(post.get("url") or "")
        text = f"{title} {body} r/{subreddit}"
        region_matches = normalize_regions(text)
        subreddit_region = SUBREDDIT_REGION.get(subreddit.lower())
        if subreddit_region and subreddit_region not in region_matches:
            region_matches.append(subreddit_region)
        issue_matches = _matched_issue_terms(text)
        relevance = _relevance_score(title, body, subreddit)
        if subreddit.lower() == "politics" and not region_matches:
            continue
        if not title:
            continue
        issue_area = _issue_area_for_matches(issue_matches)
        created_date = datetime.fromtimestamp(created, timezone.utc).date().isoformat() if created else datetime.now(timezone.utc).date().isoformat()
        readable_subreddit = f"r/{subreddit}"
        rows.append(
            {
                "date": created_date,
                "source_name": readable_subreddit,
                "source_type": "public Reddit discussion",
                "headline": title,
                "snippet": _snippet(body, post),
                "url": url,
                "geography": ", ".join(region_matches) if region_matches else "Statewide",
                "geography_refs": ", ".join(region_matches) if region_matches else "",
                "geography_matches": ", ".join(region_matches),
                "issue_area": issue_area,
                "source_platform": "reddit",
                "subreddit": subreddit,
                "score": int(post.get("score") or 0),
                "num_comments": int(post.get("num_comments") or 0),
                "relevance_score": relevance,
                "issue_keyword_matches": ", ".join(issue_matches),
                "language": "English",
            }
        )
    return rows


def _request_json(url: str, timeout_seconds: int = 20, attempts: int = 3) -> dict:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8")
            return json.loads(raw)
        except HTTPError as exc:
            last_error = exc
            if exc.code in {429, 500, 502, 503, 504} and attempt < attempts - 1:
                time.sleep(3 * (attempt + 1))
                continue
            raise
        except (URLError, json.JSONDecodeError, TimeoutError) as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"Reddit request failed: {last_error}")


def _matched_issue_terms(text: str) -> list[str]:
    lowered = text.lower()
    return [term for term in ISSUE_TERMS if term.lower() in lowered]


def _issue_area_for_matches(matches: list[str]) -> str | None:
    text = " ".join(matches)
    if not text:
        return None
    best_issue = None
    best_count = 0
    for issue, keywords in ISSUE_KEYWORDS.items():
        count = sum(1 for keyword in keywords if keyword.lower() in text.lower())
        if count > best_count:
            best_issue = issue
            best_count = count
    return best_issue


def _relevance_score(title: str, body: str, subreddit: str) -> int:
    title_regions = bool(normalize_regions(title))
    body_regions = bool(normalize_regions(body))
    title_issue = bool(_matched_issue_terms(title))
    body_issue = bool(_matched_issue_terms(body))
    ny_specific_subreddit = subreddit.lower() in NY_SPECIFIC_SUBREDDITS
    return (
        int(title_regions) * 2
        + int(body_regions)
        + int(title_issue)
        + int(body_issue)
        + int(ny_specific_subreddit)
    )


def _snippet(body: str, post: dict) -> str:
    if body:
        return body[:420]
    link = post.get("url_overridden_by_dest") or post.get("url") or ""
    return f"Reddit discussion link: {link}" if link else "Public Reddit discussion post."


def _quote(term: str) -> str:
    return f'"{term}"' if " " in term else term


def summarize_reddit_file(path: str | Path = "data/reddit_posts.csv") -> dict:
    df = pd.read_csv(path)
    failures = ""
    if not df.empty and "collection_failures" in df:
        raw_failures = df["collection_failures"].dropna()
        failures = "" if raw_failures.empty else str(raw_failures.iloc[0])
    return {
        "posts": len(df),
        "rows_before_filtering": int(df["rows_before_filtering"].iloc[0])
        if "rows_before_filtering" in df and not df.empty
        else len(df),
        "rows_after_filtering": int(df["rows_after_filtering"].iloc[0])
        if "rows_after_filtering" in df and not df.empty
        else len(df),
        "start_date": str(pd.to_datetime(df["date"]).min().date()) if not df.empty else None,
        "end_date": str(pd.to_datetime(df["date"]).max().date()) if not df.empty else None,
        "top_subreddits": df["subreddit"].value_counts().head(5).to_dict() if "subreddit" in df else {},
        "top_regions": df["geography_matches"].value_counts().head(5).to_dict() if "geography_matches" in df else {},
        "failures": failures,
    }


if __name__ == "__main__":
    result = fetch_latest_reddit_posts()
    print(summarize_reddit_file())
