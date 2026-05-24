"""Campaign research memo generation from narrative radar outputs."""

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


def generate_research_memo(df: pd.DataFrame, rollup: pd.DataFrame) -> str:
    latest_date = pd.to_datetime(df["date"]).max().date()
    previous_date = pd.to_datetime(df["date"]).min().date()
    test_issues = rollup[rollup["radar_flag"] == "test"]
    watch_issues = rollup[rollup["radar_flag"] == "watch"]

    lines = [
        "# Social Listening Research Memo",
        "",
        f"**Window:** {previous_date} to {latest_date}",
        "",
        "## What Changed",
        f"- The sample public-discourse set contains {len(df)} items across {df['classified_issue_area'].nunique()} issue areas.",
        f"- Highest-volume issue: {rollup.sort_values('mentions', ascending=False).iloc[0]['classified_issue_area']}.",
        f"- Highest-intensity issue: {rollup.sort_values('avg_intensity', ascending=False).iloc[0]['classified_issue_area']}.",
        "",
        "## Issues Spiking",
    ]

    if test_issues.empty and watch_issues.empty:
        lines.append("- No issue crosses the watch threshold in this sample window.")
    else:
        for _, row in pd.concat([test_issues, watch_issues]).iterrows():
            lines.append(
                f"- **{row['classified_issue_area']}**: {row['radar_flag']} "
                f"(spike score {row['spike_score']:.2f}, avg intensity {row['avg_intensity']:.1f})."
            )

    lines.extend(["", "## Likely Voter Concern Behind The Discourse"])
    for _, row in rollup.iterrows():
        issue = row["classified_issue_area"]
        lines.append(f"- **{issue}:** {VOTER_CONCERNS.get(issue, 'issue salience and institutional responsiveness')}.")

    lines.extend(["", "## Recommended Message Hypotheses"])
    for _, row in rollup.iterrows():
        issue = row["classified_issue_area"]
        if row["radar_flag"] in {"test", "watch"}:
            for hypothesis in MESSAGE_HYPOTHESES.get(issue, []):
                lines.append(f"- **{issue}:** {hypothesis}")

    lines.extend(
        [
            "",
            "## What Should Be Tested Next",
            "- Convert watch/test issues into candidate message arms: economic frame, values/trust frame, and competence/problem-solving frame.",
            "- Use public-discourse context features to prioritize research questions and design adaptive experiments.",
            "- Define rewards around engagement quality, surveys, signups, and other consent-based signals without claiming individual persuasion.",
            "",
            "## Limitations",
            "- Public data only; no private voter data is used.",
            "- Keyword classification is transparent but approximate.",
            "- Sentiment and intensity scores are simple research triage signals, not truth labels.",
            "- The bandit log is simulated and intended only to show future adaptive experimentation design.",
            "- Production use would require legal/privacy review, platform compliance, and experimental safeguards.",
        ]
    )
    return "\n".join(lines)
