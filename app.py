from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.bandit_simulator import (
    REWARD_SIGNALS,
    build_context_features,
    generate_message_arms,
    simulate_policies,
)
from src.classify_topics import classify_records, geography_counts, summarize_keyword_rules
from src.generate_memo import generate_research_memo
from src.scoring import add_scores, add_spike_scores, daily_issue_volume, issue_rollup

ROOT = Path(__file__).parent
ARTICLE_PATH = ROOT / "data" / "sample_articles.csv"
BANDIT_LOG_PATH = ROOT / "data" / "sample_bandit_log.csv"
MEMO_PATH = ROOT / "outputs" / "sample_research_memo.md"


st.set_page_config(
    page_title="Social Listening",
    page_icon="SL",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_articles() -> pd.DataFrame:
    raw = pd.read_csv(ARTICLE_PATH)
    classified = pd.DataFrame(classify_records(raw.to_dict(orient="records")))
    scored = add_scores(classified)
    return add_spike_scores(scored)


@st.cache_data
def load_bandit_log() -> pd.DataFrame:
    return pd.read_csv(BANDIT_LOG_PATH)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f7f8fb;
            color: #1f2937;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 16px 18px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .radar-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        }
        .story-strip {
            background: #111827;
            color: #ffffff;
            border-radius: 8px;
            padding: 18px 20px;
            margin: 14px 0 18px 0;
            font-size: 1.05rem;
            font-weight: 650;
        }
        .summary-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            min-height: 150px;
            padding: 16px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .summary-label {
            color: #4b5563;
            font-size: 0.78rem;
            font-weight: 750;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .summary-body {
            color: #111827;
            font-size: 0.98rem;
            line-height: 1.45;
        }
        .principle-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 16px;
            min-height: 184px;
        }
        .flag-test {
            color: #991b1b;
            font-weight: 700;
        }
        .flag-watch {
            color: #9a3412;
            font-weight: 700;
        }
        .flag-ignore {
            color: #166534;
            font-weight: 700;
        }
        .small-muted {
            color: #6b7280;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def chart_dataframes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    volume = daily_issue_volume(df)
    issue_mix = (
        df["classified_issue_area"].value_counts().rename_axis("issue_area").reset_index(name="mentions")
    )
    tone = (
        df.groupby(["classified_issue_area", "tone"]).size().reset_index(name="items")
    )
    return volume, issue_mix, tone


def render_header(df: pd.DataFrame, rollup: pd.DataFrame) -> None:
    st.title("Social Listening")
    st.caption(
        "Narrative intelligence prototype for campaign research."
    )
    st.markdown(
        """
        <div class="story-strip">
            Public discourse -> issue detection -> narrative monitoring -> research synthesis -> message hypotheses
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    cols[0].metric("Total mentions", f"{len(df):,}")
    cols[1].metric("Issue areas", df["classified_issue_area"].nunique())
    cols[2].metric("Watch/Test issues", int(rollup["radar_flag"].isin(["watch", "test"]).sum()))
    cols[3].metric("Latest sample", str(pd.to_datetime(df["date"]).max().date()))


def render_executive_summary(df: pd.DataFrame, rollup: pd.DataFrame) -> None:
    top_issue = rollup.sort_values(["spike_score", "avg_intensity"], ascending=False).iloc[0]
    top_geo = next(iter(geography_counts(df.to_dict(orient="records"))), "Statewide")
    test_issues = rollup[rollup["radar_flag"] == "test"]["classified_issue_area"].tolist()
    watch_issues = rollup[rollup["radar_flag"] == "watch"]["classified_issue_area"].tolist()
    next_issue = test_issues[0] if test_issues else (watch_issues[0] if watch_issues else top_issue["classified_issue_area"])

    cards = [
        (
            "What changed",
            f"{top_issue['classified_issue_area']} is carrying the strongest recent signal "
            f"(spike {top_issue['spike_score']:.2f}, intensity {top_issue['avg_intensity']:.1f}).",
        ),
        (
            "Where it matters",
            f"The sample points first to {top_geo}, with geography mentions used as research context rather than individual-level decisions.",
        ),
        (
            "What to test next",
            f"Turn {next_issue} into economic, values/trust, and competence/problem-solving message hypotheses.",
        ),
        (
            "What the system would learn",
            "Which narratives are emerging, which concerns sit underneath them, and which message hypotheses deserve testing.",
        ),
    ]

    st.write("**Executive Summary**")
    columns = st.columns(4)
    for column, (label, body) in zip(columns, cards):
        column.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">{label}</div>
                <div class="summary-body">{body}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_overview(df: pd.DataFrame, rollup: pd.DataFrame) -> None:
    volume, issue_mix, tone = chart_dataframes(df)
    st.subheader("Overview")
    st.write("A five-minute walkthrough starts here: what moved, where it matters, and what research should happen next.")
    render_executive_summary(df, rollup)
    st.divider()

    left, right = st.columns([1.3, 1])
    with left:
        st.line_chart(volume, x="date", y="mentions", color="classified_issue_area", height=320)
    with right:
        st.bar_chart(issue_mix, x="issue_area", y="mentions", height=320)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.write("**Sentiment/Tone by Issue**")
        st.dataframe(
            rollup[["classified_issue_area", "mentions", "avg_sentiment", "avg_intensity", "radar_flag"]]
            .rename(
                columns={
                    "classified_issue_area": "issue_area",
                    "avg_sentiment": "avg_sentiment_score",
                    "avg_intensity": "avg_narrative_intensity",
                }
            )
            .style.format({"avg_sentiment_score": "{:.2f}", "avg_narrative_intensity": "{:.1f}"}),
            hide_index=True,
            width="stretch",
        )
    with c2:
        geo = pd.DataFrame(
            geography_counts(df.to_dict(orient="records")).items(),
            columns=["geography", "mentions"],
        ).head(8)
        st.write("**Top NY Geographies Mentioned**")
        st.bar_chart(geo, x="geography", y="mentions", height=260)

    with st.expander("Tone count details"):
        st.dataframe(tone, hide_index=True, width="stretch")


def render_narrative_radar(df: pd.DataFrame, rollup: pd.DataFrame) -> None:
    st.subheader("Narrative Radar")
    st.write(
        "Articles and posts are classified with transparent keyword rules, then scored for tone, "
        "narrative intensity, and spikes versus a rolling baseline."
    )

    with st.expander("Keyword rules used for classification", expanded=False):
        rules = pd.DataFrame(
            [{"issue_area": issue, "sample_keywords": words} for issue, words in summarize_keyword_rules().items()]
        )
        st.dataframe(rules, hide_index=True, width="stretch")

    selected_issue = st.selectbox(
        "Issue focus",
        options=rollup["classified_issue_area"].tolist(),
    )
    selected_row = rollup[rollup["classified_issue_area"] == selected_issue].iloc[0]
    flag_class = f"flag-{selected_row['radar_flag']}"
    st.markdown(
        f"""
        <div class="radar-card">
          <div class="{flag_class}">{selected_row['radar_flag'].upper()}</div>
          <div><strong>{selected_issue}</strong></div>
          <div class="small-muted">
            Spike score {selected_row['spike_score']:.2f} |
            Avg intensity {selected_row['avg_intensity']:.1f} |
            Avg sentiment {selected_row['avg_sentiment']:.2f}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    issue_df = df[df["classified_issue_area"] == selected_issue].sort_values("date", ascending=False)
    st.write("**Top headlines and snippets**")
    st.dataframe(
        issue_df[
            [
                "date",
                "source_name",
                "headline",
                "snippet",
                "detected_geographies",
                "tone",
                "narrative_intensity",
                "spike_score",
                "keyword_hits",
            ]
        ].head(8),
        hide_index=True,
        width="stretch",
    )


def render_memo(df: pd.DataFrame, rollup: pd.DataFrame) -> None:
    st.subheader("Research Memo")
    memo = generate_research_memo(df, rollup)
    MEMO_PATH.parent.mkdir(exist_ok=True)
    MEMO_PATH.write_text(memo, encoding="utf-8")
    st.markdown(memo)


def render_bandit_readiness(df: pd.DataFrame, rollup: pd.DataFrame) -> None:
    st.subheader("Bandit Readiness: Future Extension")
    st.write(
        "This lightweight section shows how structured narrative intelligence could feed future adaptive "
        "experimentation. It is downstream of the social listening workflow, not the core product."
    )

    context_features = build_context_features(df, rollup)
    bandit_log = load_bandit_log()
    selected_issue = st.selectbox(
        "Build message arms for issue",
        options=sorted(df["classified_issue_area"].unique()),
        key="bandit_issue",
    )

    left, right = st.columns([1, 1])
    with left:
        st.write("**1. Context Features From The Radar**")
        st.caption("These are issue and discourse features, not private voter records.")
        st.dataframe(context_features, hide_index=True, width="stretch")
    with right:
        st.write("**2. Candidate Message Arms**")
        st.caption("Each arm is a testable research hypothesis, not a persuasion claim.")
        st.dataframe(generate_message_arms(selected_issue), hide_index=True, width="stretch")

    st.write("**3. Hypothetical Reward Definitions**")
    reward_cols = st.columns(3)
    for idx, signal in enumerate(REWARD_SIGNALS):
        reward_cols[idx % 3].markdown(f"- {signal}")
    st.caption(
        "Rewards are research and engagement signals. This demo does not claim real persuasion and does not optimize voters."
    )

    learn_col, ope_col = st.columns(2)
    with learn_col:
        st.markdown(
            """
            <div class="principle-card">
              <div class="summary-label">Why this preserves learning</div>
              <div class="summary-body">
                The log keeps the context, message arm, propensity score, reward, and logging policy together.
                That means future analysis can separate what was shown from why it was shown, instead of
                only chasing the current best-looking message.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ope_col:
        st.markdown(
            """
            <div class="principle-card">
              <div class="summary-label">How OPE would be used later</div>
              <div class="summary-body">
                Off-policy evaluation would use logged propensities to estimate how a new policy might have
                performed before launch. In production, that would support review, canary decisions, and
                safeguards before any adaptive policy receives more traffic.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Sample simulated bandit log", expanded=True):
        st.dataframe(bandit_log, hide_index=True, width="stretch")

    st.write("**4. Policy Simulator Output**")
    st.dataframe(simulate_policies(bandit_log), hide_index=True, width="stretch")


def render_what_is() -> None:
    st.subheader("What this is / what this is not")
    is_col, not_col = st.columns(2)
    with is_col:
        st.markdown(
            """
            <div class="principle-card">
              <div class="summary-label">What this is</div>
              <div class="summary-body">
                A public-data social listening prototype for issue detection, narrative monitoring,
                campaign research synthesis, and message hypothesis generation.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with not_col:
        st.markdown(
            """
            <div class="principle-card">
              <div class="summary-label">What this is not</div>
              <div class="summary-body">
                Not voter microtargeting. Not a production campaign platform. Not a claim of persuasion
                effects. Not a real contextual bandit deployment or a system using private voter data.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_ethics() -> None:
    st.subheader("Ethics and Limitations")
    st.markdown(
        """
        - Public data only.
        - No private voter data.
        - No individual persuasion claims.
        - No demographic microtargeting.
        - Transparent rules.
        - Simulated bandit log only.
        - Production versions would require legal/privacy review, platform compliance, and experimental safeguards.
        """
    )


def main() -> None:
    inject_css()
    df = load_articles()
    rollup = issue_rollup(df)

    with st.sidebar:
        st.header("Demo Controls")
        st.caption("Filter the sample corpus for walkthroughs.")
        issues = st.multiselect(
            "Issue areas",
            options=sorted(df["classified_issue_area"].unique()),
            default=sorted(df["classified_issue_area"].unique()),
        )
        source_types = st.multiselect(
            "Source types",
            options=sorted(df["source_type"].unique()),
            default=sorted(df["source_type"].unique()),
        )
        st.divider()
        st.caption("No API keys or internet access required.")

    filtered = df[
        df["classified_issue_area"].isin(issues)
        & df["source_type"].isin(source_types)
    ].copy()
    filtered_rollup = issue_rollup(filtered) if not filtered.empty else rollup

    render_header(filtered, filtered_rollup)
    tab_overview, tab_radar, tab_memo, tab_bandit, tab_what_is, tab_ethics = st.tabs(
        [
            "Overview",
            "Narrative Radar",
            "Research Memo",
            "Future Experimentation",
            "What this is",
            "Ethics",
        ]
    )
    with tab_overview:
        render_overview(filtered, filtered_rollup)
    with tab_radar:
        render_narrative_radar(filtered, filtered_rollup)
    with tab_memo:
        render_memo(filtered, filtered_rollup)
    with tab_bandit:
        render_bandit_readiness(filtered, filtered_rollup)
    with tab_what_is:
        render_what_is()
    with tab_ethics:
        render_ethics()


if __name__ == "__main__":
    main()
