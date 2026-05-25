"""Simple policy simulator placeholders for adaptive experimentation design."""

from __future__ import annotations

import numpy as np
import pandas as pd

MESSAGE_ARMS = {
    "economic frame": "Lead with household economics, costs, jobs, and material relief.",
    "values/trust frame": "Lead with fairness, accountability, community values, and public trust.",
    "competence/problem-solving frame": "Lead with delivery, operational competence, and measurable fixes.",
}

REWARD_SIGNALS = [
    "email click",
    "donation conversion",
    "volunteer signup",
    "survey persuasion lift",
    "canvass response",
    "content engagement quality",
]


def generate_message_arms(issue_area: str) -> pd.DataFrame:
    rows = []
    for arm, description in MESSAGE_ARMS.items():
        rows.append(
            {
                "issue_area": issue_area,
                "message_arm": arm,
                "hypothesis": f"For {issue_area}, {description.lower()}",
            }
        )
    return pd.DataFrame(rows)


def simulate_policies(log_df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    arms = sorted(log_df["message_arm"].unique())
    observed_means = log_df.groupby("message_arm")["reward"].mean().to_dict()
    global_mean = float(log_df["reward"].mean())

    results = []
    for policy in ["random policy", "epsilon-greedy placeholder", "Thompson sampling placeholder"]:
        if policy == "random policy":
            chosen = rng.choice(arms, size=len(log_df), replace=True)
            expected = np.mean([observed_means.get(arm, global_mean) for arm in chosen])
            notes = "Uniform exploration across message arms."
        elif policy == "epsilon-greedy placeholder":
            best_arm = max(observed_means, key=observed_means.get)
            explore = rng.random(len(log_df)) < 0.15
            chosen = np.where(explore, rng.choice(arms, size=len(log_df), replace=True), best_arm)
            expected = np.mean([observed_means.get(arm, global_mean) for arm in chosen])
            notes = "Mostly uses the currently best observed arm while reserving exploration."
        else:
            sampled_values = {
                arm: rng.beta(
                    1 + log_df[(log_df["message_arm"] == arm) & (log_df["reward"] > global_mean)].shape[0],
                    1 + log_df[(log_df["message_arm"] == arm) & (log_df["reward"] <= global_mean)].shape[0],
                )
                for arm in arms
            }
            best_arm = max(sampled_values, key=sampled_values.get)
            expected = observed_means.get(best_arm, global_mean)
            notes = "Illustrative Bayesian sampling placeholder; not a production estimator."

        results.append(
            {
                "policy": policy,
                "simulated_expected_reward": round(float(expected), 3),
                "exploration_rate": 1.0 if policy == "random policy" else (0.15 if "epsilon" in policy else 0.25),
                "notes": notes,
            }
        )
    return pd.DataFrame(results)


def build_context_features(radar_df: pd.DataFrame, rollup_df: pd.DataFrame) -> pd.DataFrame:
    latest_by_issue = radar_df.sort_values("date").groupby("classified_issue_area").tail(1)
    geography_column = "region" if "region" in latest_by_issue.columns else "detected_geographies"
    merged = latest_by_issue.merge(
        rollup_df[["classified_issue_area", "urgency_level"]],
        on="classified_issue_area",
        how="left",
    )
    return merged[
        [
            "classified_issue_area",
            geography_column,
            "source_type",
            "sentiment_score",
            "narrative_intensity",
            "spike_score",
            "audience_segment_hypothesis",
            "urgency_level",
        ]
    ].rename(
        columns={
            "classified_issue_area": "issue_area",
            geography_column: "region",
            "narrative_intensity": "attention_and_tone",
            "spike_score": "change_vs_recent_baseline",
        }
    )
