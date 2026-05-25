"""Generate an operational-scale demo corpus from observed NY discourse patterns."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


ISSUE_NARRATIVES = {
    "affordability / cost of living": [
        "families describe grocery and utility pressure",
        "commuters connect fare pressure to household budgets",
        "local officials debate affordability relief proposals",
        "residents ask for concrete cost-of-living plans",
    ],
    "housing / rent": [
        "renters describe instability and repair delays",
        "housing advocates warn that eviction pressure is rising",
        "local leaders debate delivery of affordable units",
        "tenants connect rent increases to neighborhood displacement",
    ],
    "immigration / public safety": [
        "residents link shelter planning to neighborhood quality of life",
        "officials debate public safety coordination and fairness",
        "community groups ask for clearer asylum support plans",
        "subway and downtown safety concerns continue to circulate",
    ],
    "AI / tech jobs": [
        "workers ask whether AI investment will produce durable jobs",
        "regional employers highlight training needs around automation",
        "tech expansion stories raise questions about workforce stability",
        "labor groups call for guardrails around artificial intelligence",
    ],
    "corruption / competence / trust": [
        "watchdogs press for accountability and procurement transparency",
        "voters connect ethics stories to broader competence concerns",
        "local coverage questions whether government can deliver reliably",
        "reform groups frame trust as a campaign issue",
    ],
}

GEOGRAPHIES = ["NYC", "Long Island", "Hudson Valley", "Capital Region", "Central NY", "Western NY"]

GEO_DETAILS = {
    "NYC": ["Queens", "Brooklyn", "Bronx", "Manhattan", "Staten Island"],
    "Long Island": ["Nassau", "Suffolk", "Hempstead", "Huntington"],
    "Hudson Valley": ["Westchester", "Rockland", "Yonkers", "Orange County"],
    "Capital Region": ["Albany", "Schenectady", "Troy", "Saratoga"],
    "Central NY": ["Syracuse", "Utica", "Rome"],
    "Western NY": ["Buffalo", "Rochester", "Erie County", "Monroe County"],
}

SOURCE_TYPES = [
    "local news",
    "public radio",
    "community forum",
    "social listening digest",
    "policy newsletter",
    "TV transcript",
]

SOURCE_NAMES = [
    "NY Civic Monitor",
    "Empire State Ledger",
    "Metro Narrative Scan",
    "Hudson Valley Public Radio",
    "Long Island Observer",
    "Buffalo Civic Wire",
    "Statehouse Watch",
    "Queens Community Post",
    "Rochester Herald",
    "Syracuse Policy Review",
]

TONE_WORDS = {
    "concerned": ["pressure", "concern", "strain", "frustration", "urgent"],
    "mixed": ["debate", "questions", "discussion", "attention", "response"],
    "constructive": ["solution", "progress", "training", "investment", "accountability"],
}


def generate_operational_demo_corpus(
    seed_path: str | Path = "data/gdelt_articles.csv",
    sample_path: str | Path = "data/sample_articles.csv",
    output_path: str | Path = "data/operational_demo_corpus.csv",
    n_rows: int = 2600,
    random_seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    seed_df = _load_seed(seed_path, sample_path)
    issue_weights = seed_df["issue_area"].value_counts(normalize=True).to_dict()
    if not issue_weights:
        issue_weights = {issue: 1 / len(ISSUE_NARRATIVES) for issue in ISSUE_NARRATIVES}

    source_weights = seed_df.get("source_type", pd.Series(dtype=str)).value_counts(normalize=True).to_dict()
    latest = pd.to_datetime(seed_df["date"], errors="coerce").max()
    if pd.isna(latest):
        latest = pd.Timestamp.today().normalize()

    rows: list[dict] = []
    issues = list(ISSUE_NARRATIVES)
    issue_probs = np.array([issue_weights.get(issue, 0.08) for issue in issues], dtype=float)
    issue_probs = issue_probs / issue_probs.sum()

    for idx in range(n_rows):
        issue = str(rng.choice(issues, p=issue_probs))
        geography = _weighted_geography(issue, rng)
        locality = str(rng.choice(GEO_DETAILS[geography]))
        tone = str(rng.choice(["concerned", "mixed", "constructive"], p=[0.46, 0.38, 0.16]))
        narrative = str(rng.choice(ISSUE_NARRATIVES[issue]))
        tone_word = str(rng.choice(TONE_WORDS[tone]))
        days_ago = int(rng.integers(0, 30))
        hour = int(rng.integers(5, 23))
        minute = int(rng.integers(0, 59))
        timestamp = latest - timedelta(days=days_ago, hours=hour, minutes=minute)
        source_type = _weighted_source_type(source_weights, rng)
        source_name = str(rng.choice(SOURCE_NAMES))
        headline = f"{locality} {narrative} as {tone_word} grows"
        snippet = (
            f"{source_name} notes that {issue} narratives in {geography} are being repeated "
            f"across {source_type} coverage, with {locality} residents emphasizing {tone_word}."
        )
        rows.append(
            {
                "date": timestamp.date().isoformat(),
                "timestamp": timestamp.isoformat(),
                "source_name": source_name,
                "headline": headline,
                "snippet": snippet,
                "url": f"https://example.com/operational-demo/{idx:05d}",
                "issue_area": issue,
                "geography_refs": f"{locality}; {geography}",
                "geography_matches": f"{locality}; {geography}",
                "source_type": source_type,
                "domain": _domain_for_source(source_name),
                "language": "English",
                "is_synthetic_operational_demo": True,
            }
        )

    out = pd.DataFrame(rows).sort_values(["date", "timestamp"], ascending=False).reset_index(drop=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    return out


def _load_seed(seed_path: str | Path, sample_path: str | Path) -> pd.DataFrame:
    seed = Path(seed_path)
    if seed.exists():
        return pd.read_csv(seed)
    return pd.read_csv(sample_path)


def _weighted_geography(issue: str, rng: np.random.Generator) -> str:
    if issue in {"affordability / cost of living", "AI / tech jobs"}:
        probs = [0.34, 0.23, 0.13, 0.10, 0.08, 0.12]
    elif issue == "housing / rent":
        probs = [0.44, 0.19, 0.18, 0.07, 0.05, 0.07]
    elif issue == "immigration / public safety":
        probs = [0.42, 0.20, 0.11, 0.08, 0.06, 0.13]
    else:
        probs = [0.26, 0.17, 0.13, 0.23, 0.08, 0.13]
    return str(rng.choice(GEOGRAPHIES, p=np.array(probs) / sum(probs)))


def _weighted_source_type(source_weights: dict[str, float], rng: np.random.Generator) -> str:
    if source_weights and len(source_weights) > 1:
        keys = list(source_weights)
        probs = np.array([source_weights[key] for key in keys], dtype=float)
        return str(rng.choice(keys, p=probs / probs.sum()))
    return str(rng.choice(SOURCE_TYPES))


def _domain_for_source(source_name: str) -> str:
    return source_name.lower().replace(" ", "").replace(".", "") + ".example"


if __name__ == "__main__":
    df = generate_operational_demo_corpus()
    print(f"Wrote {len(df)} rows to data/operational_demo_corpus.csv")
