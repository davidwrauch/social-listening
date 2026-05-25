"""Shared New York region normalization."""

from __future__ import annotations

import re


NY_REGIONS = [
    "NYC",
    "Long Island",
    "Hudson Valley",
    "Capital Region",
    "Central NY",
    "Western NY",
]

REGION_KEYWORDS: dict[str, list[str]] = {
    "NYC": [
        "nyc",
        "new york city",
        "manhattan",
        "brooklyn",
        "queens",
        "bronx",
        "staten island",
    ],
    "Long Island": ["long island", "nassau", "suffolk", "hempstead", "huntington"],
    "Hudson Valley": ["hudson valley", "westchester", "rockland", "orange county", "yonkers"],
    "Capital Region": ["capital region", "albany", "schenectady", "troy", "saratoga"],
    "Central NY": ["central ny", "central new york", "syracuse", "utica", "rome"],
    "Western NY": ["western ny", "western new york", "buffalo", "rochester", "erie county", "monroe county"],
}

NY_GEOGRAPHY_TERMS = sorted({term for terms in REGION_KEYWORDS.values() for term in terms} | {"new york"})


def normalize_regions(text: object) -> list[str]:
    normalized = re.sub(r"\s+", " ", str(text).lower())
    matches: list[str] = []
    for region, keywords in REGION_KEYWORDS.items():
        if any(_contains_term(normalized, keyword) for keyword in keywords):
            matches.append(region)
    if not matches and _contains_term(normalized, "new york"):
        matches.append("NYC")
    return matches


def normalize_region_label(text: object) -> str:
    matches = normalize_regions(text)
    return ", ".join(matches) if matches else "Statewide"


def _contains_term(text: str, term: str) -> bool:
    return re.search(r"\b" + re.escape(term.lower()) + r"\b", text) is not None
