"""Campaign research memo generation from scored public-discourse signals."""

from __future__ import annotations

import pandas as pd


MESSAGE_HYPOTHESES = {
    "affordability / cost of living": [
        "Connect cost pressure to concrete household relief and fiscal discipline.",
        "Test whether local examples of lower everyday costs outperform generic affordability language.",
    ],
    "housing / rent": [
        "Frame housing as stability, neighborhood continuity, and faster delivery of units.",
        "Test problem-solving language on permitting, vacancies, and tenant protections.",
    ],
    "immigration / public safety": [
        "Separate compassion, order, and quality-of-life concerns in transparent language.",
        "Test whether operational competence language lowers heat around safety narratives.",
    ],
    "AI / tech jobs": [
        "Emphasize worker readiness, regional opportunity, and guardrails for automation.",
        "Test jobs-and-training language against broader innovation competitiveness framing.",
    ],
    "corruption / competence / trust": [
        "Lead with accountability, procurement clarity, and measurable delivery standards.",
        "Test a trust-and-results message against a stronger anti-corruption contrast.",
    ],
}

VOTER_CONCERNS = {
    "affordability / cost of living": "households feeling squeezed by prices, taxes, utilities, and daily expenses",
    "housing / rent": "renters and families worried that stable housing is moving out of reach",
    "immigration / public safety": "residents looking for order, fairness, and visible quality-of-life management",
    "AI / tech jobs": "workers asking whether economic change will create opportunity or displacement",
    "corruption / competence / trust": "voters questioning whether institutions can deliver honestly and competently",
}


def format_issue(issue: str) -> str:
    if issue.startswith("AI"):
        return "AI" + issue[2:].capitalize()
    return issue.capitalize()


def generate_research_memo(df: pd.DataFrame, rollup: pd.DataFrame) -> str:
    latest_date = pd.to_datetime(df["date"]).max().date()
    previous_date = pd.to_datetime(df["date"]).min().date()
    test_issues = rollup[rollup["radar_flag"] == "test"]
    watch_issues = rollup[rollup["radar_flag"] == "watch"]
    rising = rollup.sort_values("spike_score", ascending=False).head(3)
    most_negative = rollup.sort_values("avg_sentiment", ascending=True).iloc[0]
    strongest_attention = rollup.sort_values("avg_intensity", ascending=False).iloc[0]

    lines = [
        "# Social Listening Research Memo",
        "",
        f"**Stories analyzed:** {len(df)} public stories and posts from {previous_date} to {latest_date}.",
        "",
        "**Attention and tone:** how much attention an issue is receiving and whether coverage is mostly positive, negative, or mixed.",
        "",
        "## Biggest shifts in discussion",
        f"- **Fastest-rising discussion:** {format_issue(rising.iloc[0]['classified_issue_area'])}; {_volume_phrase(rising.iloc[0]['spike_score'])}.",
        f"- **Most sustained attention:** {format_issue(rollup.sort_values('mentions', ascending=False).iloc[0]['classified_issue_area'])} has the highest story volume.",
        f"- **Most negative tone:** {format_issue(most_negative['classified_issue_area'])} has the most concerned coverage language.",
        f"- **Most charged discussion:** {format_issue(strongest_attention['classified_issue_area'])} is generating sustained attention with more emotional coverage.",
        "",
        "## Issues with rising attention",
    ]

    if test_issues.empty and watch_issues.empty:
        lines.append("- No issue crosses the watch threshold in this set of stories.")
    else:
        for _, row in pd.concat([test_issues, watch_issues]).iterrows():
            lines.append(
                f"- **{format_issue(row['classified_issue_area'])}**: {str(row['radar_flag']).capitalize()} "
                f"because {_volume_phrase(row['spike_score'])} and {_attention_phrase(row['avg_intensity'])}."
            )

    lines.extend(["", "## Likely concern behind the discussion"])
    for _, row in rollup.iterrows():
        issue = row["classified_issue_area"]
        lines.append(f"- **{format_issue(issue)}:** {VOTER_CONCERNS.get(issue, 'issue salience and institutional responsiveness')}.")

    lines.extend(["", "## Recommended message hypotheses"])
    for _, row in rollup.iterrows():
        issue = row["classified_issue_area"]
        if row["radar_flag"] in {"test", "watch"}:
            for hypothesis in MESSAGE_HYPOTHESES.get(issue, []):
                lines.append(f"- **{format_issue(issue)}:** {hypothesis}")

    lines.extend(
        [
            "",
            "## What should be tested next",
            "- Convert watch/test issues into candidate message frames: economic frame, values/trust frame, and competence/problem-solving frame.",
            "- Use public-discourse context features to prioritize research questions and design adaptive experiments.",
            "- Define rewards around engagement quality, surveys, signups, and other consent-based measures without claiming individual persuasion.",
            "",
            "## Limitations",
            "- Public data only; no private voter data is used.",
            "- Keyword classification is transparent but approximate.",
            "- Tone and attention readings are simple research triage inputs, not truth labels.",
            "- The experiment log is simulated and intended only to show future adaptive experimentation design.",
            "- Production use would require legal/privacy review, platform compliance, and experimental safeguards.",
        ]
    )
    return "\n".join(lines)


def _volume_phrase(score: float) -> str:
    if score <= 0:
        return "discussion volume is similar to the recent average"
    return f"discussion volume is about {score * 100:.0f}% higher than the recent average"


def _attention_phrase(score: float) -> str:
    if score >= 5:
        return "coverage is highly emotional and drawing sustained attention"
    if score >= 3:
        return "stories are drawing steady attention with mixed or concerned tone"
    return "coverage is present but less emotionally charged"
