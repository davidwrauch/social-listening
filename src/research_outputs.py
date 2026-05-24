"""Human-readable campaign research outputs from scored narrative data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.generate_memo import MESSAGE_HYPOTHESES, VOTER_CONCERNS


FRAME_COPY = {
    "economic frame": "Connect the issue to household costs, jobs, and material stability.",
    "competence frame": "Emphasize practical problem-solving, delivery, and visible management.",
    "values/trust frame": "Lead with fairness, accountability, safety, and public trust.",
}

CHANNEL_BY_ISSUE = {
    "affordability / cost of living": "email subject-line test or survey prompt",
    "housing / rent": "focus group prompt with renter/homeowner split",
    "immigration / public safety": "moderated message test with qualitative follow-up",
    "AI / tech jobs": "digital content test with worker-opportunity language",
    "corruption / competence / trust": "survey experiment on trust and delivery claims",
}


def split_values(value: object) -> list[str]:
    if pd.isna(value):
        return []
    raw = str(value).replace(",", ";")
    return [part.strip() for part in raw.split(";") if part.strip()]


def row_geographies(row: pd.Series) -> list[str]:
    for column in ["geography_matches", "detected_geographies", "geography_refs"]:
        if column in row and split_values(row[column]):
            return split_values(row[column])
    return ["Statewide"]


def source_column(df: pd.DataFrame) -> str:
    return "domain" if "domain" in df.columns else "source_name"


def add_geo_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in df.iterrows():
        for geography in row_geographies(row):
            enriched = row.to_dict()
            enriched["geography"] = geography
            rows.append(enriched)
    return pd.DataFrame(rows)


def recommendation_for(flag: str, spike_score: float, sentiment: float) -> str:
    if flag == "test" or spike_score >= 1.0:
        return "test"
    if sentiment <= -0.35:
        return "escalate"
    return "monitor"


def generate_weekly_issue_brief(df: pd.DataFrame, rollup: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    geo_df = add_geo_rows(df)
    source_col = source_column(df)
    rows: list[dict] = []
    lines = ["# Weekly Issue Brief", ""]

    for _, issue_row in rollup.sort_values(["mentions", "spike_score"], ascending=False).iterrows():
        issue = issue_row["classified_issue_area"]
        issue_df = df[df["classified_issue_area"] == issue]
        issue_geo = geo_df[geo_df["classified_issue_area"] == issue]
        top_geographies = ", ".join(issue_geo["geography"].value_counts().head(3).index.tolist()) or "Statewide"
        top_sources = ", ".join(issue_df[source_col].dropna().astype(str).value_counts().head(3).index.tolist())
        sentiment = float(issue_row["avg_sentiment"])
        spike = float(issue_row["spike_score"])
        recommendation = recommendation_for(issue_row["radar_flag"], spike, sentiment)
        interpretation = (
            f"{issue} is drawing {int(issue_row['mentions'])} mentions, with strongest signal in "
            f"{top_geographies} and a {issue_row['radar_flag']} research priority."
        )
        row = {
            "issue_area": issue,
            "current_volume": int(issue_row["mentions"]),
            "sentiment_tone": _tone_label(sentiment),
            "change_vs_recent_baseline": round(spike, 2),
            "top_geographies": top_geographies,
            "top_sources": top_sources,
            "interpretation": interpretation,
            "recommendation": recommendation,
        }
        rows.append(row)
        lines.extend(
            [
                f"## {issue}",
                f"- Current volume: {row['current_volume']}",
                f"- Sentiment/tone: {row['sentiment_tone']}",
                f"- Change vs recent baseline: {_baseline_phrase(row['change_vs_recent_baseline'])}",
                f"- Top geographies: {top_geographies}",
                f"- Top sources: {top_sources}",
                f"- Interpretation: {interpretation}",
                f"- Recommendation: {recommendation}",
                "",
            ]
        )

    return "\n".join(lines), pd.DataFrame(rows)


def generate_geography_watchlist(df: pd.DataFrame) -> pd.DataFrame:
    geo_df = add_geo_rows(df)
    grouped = (
        geo_df.groupby(["geography", "classified_issue_area"])
        .agg(
            mentions=("headline", "count"),
            avg_sentiment=("sentiment_score", "mean"),
            avg_intensity=("narrative_intensity", "mean"),
            spike_score=("spike_score", "max"),
        )
        .reset_index()
    )
    if grouped.empty:
        return pd.DataFrame(
            columns=[
                "geography",
                "top_issue",
                "mentions",
                "avg_sentiment",
                "spike_score",
                "why_it_matters",
                "recommended_next_step",
            ]
        )

    idx = grouped.sort_values(["mentions", "avg_intensity", "spike_score"], ascending=False).groupby("geography").head(1).index
    watchlist = grouped.loc[idx].copy()
    watchlist["why_it_matters"] = watchlist.apply(
        lambda row: (
            f"{row['classified_issue_area']} is the leading signal here, with "
            f"{int(row['mentions'])} stories and discussion about {row['spike_score']:.2f}x above recent baseline."
        ),
        axis=1,
    )
    watchlist["recommended_next_step"] = watchlist.apply(
        lambda row: "escalate analyst review"
        if row["avg_sentiment"] <= -0.35
        else ("test message hypothesis" if row["spike_score"] >= 1.0 else "monitor next cycle"),
        axis=1,
    )
    watchlist = watchlist.rename(columns={"classified_issue_area": "top_issue"})
    watchlist["avg_sentiment"] = watchlist["avg_sentiment"].round(2)
    watchlist["spike_score"] = watchlist["spike_score"].round(2)
    watchlist = watchlist.rename(columns={"spike_score": "change_vs_recent_baseline"})
    return watchlist[
        [
            "geography",
            "top_issue",
            "mentions",
            "avg_sentiment",
            "change_vs_recent_baseline",
            "why_it_matters",
            "recommended_next_step",
        ]
    ].sort_values(["mentions", "change_vs_recent_baseline"], ascending=False)


def generate_message_hypothesis_bank(df: pd.DataFrame, max_pairs: int = 12) -> pd.DataFrame:
    geo_df = add_geo_rows(df)
    pairs = (
        geo_df.groupby(["classified_issue_area", "geography"])
        .agg(mentions=("headline", "count"), avg_intensity=("narrative_intensity", "mean"))
        .reset_index()
        .sort_values(["mentions", "avg_intensity"], ascending=False)
        .head(max_pairs)
    )
    rows: list[dict] = []
    for _, pair in pairs.iterrows():
        issue = pair["classified_issue_area"]
        geography = pair["geography"]
        concern = VOTER_CONCERNS.get(issue, "issue salience and institutional responsiveness")
        rows.append(
            {
                "issue_area": issue,
                "geography": geography,
                "economic_frame": f"In {geography}, connect {issue} to concrete household and economic stakes.",
                "competence_frame": f"Show a practical plan for managing {issue} with measurable delivery.",
                "values_trust_frame": f"Frame {issue} around fairness, accountability, and community trust.",
                "recommended_audience_hypothesis": concern,
                "suggested_test_channel": CHANNEL_BY_ISSUE.get(issue, "survey prompt or moderated discussion"),
                "risk_or_caveat": "Validate with human analyst review; public discourse is not a measure of persuasion.",
            }
        )
    return pd.DataFrame(rows)


def generate_research_questions(df: pd.DataFrame, rollup: pd.DataFrame, limit: int = 8) -> str:
    geo_df = add_geo_rows(df)
    questions: list[str] = []
    top_issues = rollup.sort_values(["mentions", "spike_score"], ascending=False).head(4)

    for _, row in top_issues.iterrows():
        issue = row["classified_issue_area"]
        issue_geo = geo_df[geo_df["classified_issue_area"] == issue]
        geography = issue_geo["geography"].value_counts().index[0] if not issue_geo.empty else "New York"
        concern = VOTER_CONCERNS.get(issue, "current public concern")
        questions.append(f"In {geography}, what specific concern is driving attention to {issue}: {concern}?")
        questions.append(f"Which frame for {issue} feels more credible: economic stakes, competence, or values/trust?")

    questions.append("Which narratives are voters hearing repeatedly, and which feel like isolated news-cycle noise?")
    questions.append("What evidence would make a proposed response feel concrete rather than performative?")
    deduped = list(dict.fromkeys(questions))[:limit]
    lines = ["# Polling / Focus Group Questions", ""]
    lines.extend(f"{idx}. {question}" for idx, question in enumerate(deduped, start=1))
    return "\n".join(lines)


def export_research_outputs(
    df: pd.DataFrame,
    rollup: pd.DataFrame,
    output_dir: str | Path = "outputs",
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    brief_md, brief_df = generate_weekly_issue_brief(df, rollup)
    watchlist = generate_geography_watchlist(df)
    hypothesis_bank = generate_message_hypothesis_bank(df)
    questions_md = generate_research_questions(df, rollup)

    (output / "weekly_issue_brief.md").write_text(brief_md, encoding="utf-8")
    watchlist.to_csv(output / "geography_watchlist.csv", index=False)
    hypothesis_bank.to_csv(output / "message_hypothesis_bank.csv", index=False)
    (output / "research_questions.md").write_text(questions_md, encoding="utf-8")

    return {
        "weekly_issue_brief_md": brief_md,
        "weekly_issue_brief_table": brief_df,
        "geography_watchlist": watchlist,
        "message_hypothesis_bank": hypothesis_bank,
        "research_questions_md": questions_md,
        "paths": {
            "weekly_issue_brief": output / "weekly_issue_brief.md",
            "geography_watchlist": output / "geography_watchlist.csv",
            "message_hypothesis_bank": output / "message_hypothesis_bank.csv",
            "research_questions": output / "research_questions.md",
        },
    }


def _tone_label(sentiment: float) -> str:
    if sentiment <= -0.25:
        return f"concerned ({sentiment:.2f})"
    if sentiment >= 0.25:
        return f"constructive ({sentiment:.2f})"
    return f"mixed ({sentiment:.2f})"


def _baseline_phrase(score: float) -> str:
    if score <= 0:
        return "near recent baseline"
    return f"about {score * 100:.0f}% above recent baseline"
