"""Transparent keyword topic classification for NY Narrative Radar."""

from __future__ import annotations

import re
from collections import defaultdict

ISSUE_KEYWORDS: dict[str, list[str]] = {
    "affordability / cost of living": [
        "affordability",
        "cost of living",
        "prices",
        "grocery",
        "groceries",
        "utility",
        "coned",
        "child care",
        "health care costs",
        "taxes",
        "inflation",
        "commute",
        "mta fare",
    ],
    "housing / rent": [
        "housing",
        "rent",
        "landlord",
        "eviction",
        "tenant",
        "zoning",
        "vacancy",
        "shelter",
        "affordable units",
        "rent-stabilized",
    ],
    "immigration / public safety": [
        "immigration",
        "migrant",
        "asylum",
        "border",
        "public safety",
        "crime",
        "police",
        "subway safety",
        "shoplifting",
        "quality of life",
    ],
    "AI / tech jobs": [
        "ai",
        "artificial intelligence",
        "automation",
        "tech jobs",
        "semiconductor",
        "chips",
        "startup",
        "upstate tech",
        "data center",
        "workforce training",
    ],
    "corruption / competence / trust": [
        "corruption",
        "ethics",
        "competence",
        "trust",
        "procurement",
        "patronage",
        "investigation",
        "indictment",
        "mismanagement",
        "accountability",
        "transparency",
    ],
}

GEOGRAPHY_KEYWORDS: dict[str, list[str]] = {
    "NYC": ["new york city", "nyc", "manhattan", "brooklyn", "queens", "bronx", "staten island"],
    "Long Island": ["long island", "nassau", "suffolk", "hempstead", "huntington"],
    "Hudson Valley": ["hudson valley", "westchester", "rockland", "orange county", "yonkers"],
    "Capital Region": ["capital region", "albany", "schenectady", "troy", "saratoga"],
    "Central NY": ["central ny", "syracuse", "utica", "rome"],
    "Western NY": ["western ny", "buffalo", "rochester", "erie county", "monroe county"],
    "North Country": ["north country", "watertown", "plattsburgh", "st. lawrence"],
}

AUDIENCE_SEGMENT_HINTS: dict[str, str] = {
    "affordability / cost of living": "cost-pressured households",
    "housing / rent": "renters and housing-cost concerned families",
    "immigration / public safety": "quality-of-life and community safety voters",
    "AI / tech jobs": "workers weighing economic transition and opportunity",
    "corruption / competence / trust": "accountability-focused swing persuadables",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def keyword_hits(text: str, keywords: list[str]) -> list[str]:
    normalized = _normalize(text)
    hits: list[str] = []
    for keyword in keywords:
        pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
        if re.search(pattern, normalized):
            hits.append(keyword)
    return hits


def classify_issue(headline: str, snippet: str, fallback: str | None = None) -> tuple[str, list[str]]:
    text = f"{headline} {snippet}"
    scores: dict[str, list[str]] = {}
    for issue, keywords in ISSUE_KEYWORDS.items():
        hits = keyword_hits(text, keywords)
        if hits:
            scores[issue] = hits

    if scores:
        issue = max(scores, key=lambda key: (len(scores[key]), -list(ISSUE_KEYWORDS).index(key)))
        return issue, scores[issue]

    if fallback and fallback in ISSUE_KEYWORDS:
        return fallback, ["sample label fallback"]

    return "affordability / cost of living", ["default fallback"]


def extract_geographies(text: str) -> list[str]:
    found: list[str] = []
    for geography, keywords in GEOGRAPHY_KEYWORDS.items():
        if keyword_hits(text, keywords):
            found.append(geography)
    return found or ["Statewide"]


def summarize_keyword_rules() -> dict[str, str]:
    return {issue: ", ".join(words[:8]) for issue, words in ISSUE_KEYWORDS.items()}


def issue_segment(issue_area: str) -> str:
    return AUDIENCE_SEGMENT_HINTS.get(issue_area, "public issue-engaged audiences")


def classify_records(records) -> list[dict]:
    classified = []
    for record in records:
        issue, hits = classify_issue(
            record.get("headline", ""),
            record.get("snippet", ""),
            record.get("issue_area", None),
        )
        geographies = extract_geographies(
            f"{record.get('headline', '')} {record.get('snippet', '')} {record.get('geography_refs', '')}"
        )
        enriched = dict(record)
        enriched["classified_issue_area"] = issue
        enriched["keyword_hits"] = ", ".join(hits)
        enriched["detected_geographies"] = ", ".join(geographies)
        enriched["audience_segment_hypothesis"] = issue_segment(issue)
        classified.append(enriched)
    return classified


def geography_counts(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        geographies = str(record.get("detected_geographies", "Statewide")).split(", ")
        for geography in geographies:
            counts[geography] += 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
