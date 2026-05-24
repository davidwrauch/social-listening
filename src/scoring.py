"""Explainable tone, intensity, and spike scoring."""

from __future__ import annotations

import pandas as pd

NEGATIVE_TERMS = {
    "crisis",
    "frustration",
    "anger",
    "concern",
    "fear",
    "unsafe",
    "surge",
    "pressure",
    "strain",
    "scandal",
    "failed",
    "eviction",
    "overwhelmed",
    "complaints",
    "audit",
}

POSITIVE_TERMS = {
    "relief",
    "progress",
    "solution",
    "investment",
    "jobs",
    "training",
    "accountability",
    "stabilize",
    "improve",
    "expansion",
    "agreement",
    "opened",
}

URGENCY_TERMS = {
    "now",
    "urgent",
    "deadline",
    "surge",
    "spike",
    "emergency",
    "protest",
    "audit",
    "investigation",
    "eviction",
    "fare hike",
}


def score_sentiment(text: str) -> tuple[float, str, str]:
    lowered = str(text).lower()
    negative_hits = sorted(term for term in NEGATIVE_TERMS if term in lowered)
    positive_hits = sorted(term for term in POSITIVE_TERMS if term in lowered)
    raw = len(positive_hits) - len(negative_hits)
    score = max(-1.0, min(1.0, raw / 4))
    if score <= -0.25:
        tone = "concerned"
    elif score >= 0.25:
        tone = "constructive"
    else:
        tone = "mixed"
    explanation = f"+{len(positive_hits)} positive / -{len(negative_hits)} concern terms"
    return score, tone, explanation


def score_urgency(text: str) -> int:
    lowered = str(text).lower()
    return sum(1 for term in URGENCY_TERMS if term in lowered)


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    text = scored["headline"].fillna("") + " " + scored["snippet"].fillna("")
    sentiment_results = text.apply(score_sentiment)
    scored["sentiment_score"] = sentiment_results.apply(lambda value: value[0])
    scored["tone"] = sentiment_results.apply(lambda value: value[1])
    scored["tone_explanation"] = sentiment_results.apply(lambda value: value[2])
    scored["urgency_hits"] = text.apply(score_urgency)
    scored["narrative_intensity"] = (
        1
        + scored["keyword_hits"].fillna("").apply(lambda value: 0 if not value else len(str(value).split(",")))
        + scored["urgency_hits"]
        + scored["source_type"].fillna("").str.contains("social|forum", case=False, regex=True).astype(int)
    )
    return scored


def add_spike_scores(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    daily = (
        data.groupby(["classified_issue_area", "date"])
        .agg(mentions=("headline", "count"), intensity=("narrative_intensity", "sum"))
        .reset_index()
        .sort_values(["classified_issue_area", "date"])
    )
    daily["baseline"] = daily.groupby("classified_issue_area")["mentions"].transform(
        lambda series: series.shift(1).rolling(3, min_periods=1).mean()
    )
    daily["baseline"] = daily["baseline"].fillna(1.0)
    daily["spike_score"] = ((daily["mentions"] - daily["baseline"]) / daily["baseline"]).clip(lower=0)

    issue_spikes = (
        daily.sort_values("date")
        .groupby("classified_issue_area")
        .tail(1)[["classified_issue_area", "spike_score"]]
    )
    latest_spike = dict(zip(issue_spikes["classified_issue_area"], issue_spikes["spike_score"]))
    data["spike_score"] = data["classified_issue_area"].map(latest_spike).fillna(0.0)
    return data


def issue_rollup(df: pd.DataFrame) -> pd.DataFrame:
    rollup = (
        df.groupby("classified_issue_area")
        .agg(
            mentions=("headline", "count"),
            avg_sentiment=("sentiment_score", "mean"),
            avg_intensity=("narrative_intensity", "mean"),
            spike_score=("spike_score", "max"),
        )
        .reset_index()
    )
    rollup["radar_flag"] = rollup.apply(flag_issue, axis=1)
    rollup["urgency_level"] = rollup.apply(urgency_level, axis=1)
    return rollup.sort_values(["radar_flag", "spike_score", "avg_intensity"], ascending=[False, False, False])


def flag_issue(row: pd.Series) -> str:
    if row["spike_score"] >= 0.75 or row["avg_intensity"] >= 5:
        return "test"
    if row["spike_score"] >= 0.25 or row["mentions"] >= 3:
        return "watch"
    return "ignore"


def urgency_level(row: pd.Series) -> str:
    if row["radar_flag"] == "test":
        return "high"
    if row["radar_flag"] == "watch":
        return "medium"
    return "low"


def daily_issue_volume(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(date=pd.to_datetime(df["date"]).dt.date)
        .groupby(["date", "classified_issue_area"])
        .size()
        .reset_index(name="mentions")
    )
