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
from src.collect_gdelt import fetch_latest_gdelt_articles
from src.generate_memo import generate_research_memo
from src.research_outputs import export_research_outputs
from src.scoring import add_scores, add_spike_scores, daily_issue_volume, issue_rollup
from src.synthetic_corpus import generate_operational_demo_corpus

ROOT = Path(__file__).parent
ARTICLE_PATH = ROOT / "data" / "sample_articles.csv"
GDELT_PATH = ROOT / "data" / "gdelt_articles.csv"
OPERATIONAL_PATH = ROOT / "data" / "operational_demo_corpus.csv"
BANDIT_LOG_PATH = ROOT / "data" / "sample_bandit_log.csv"
MEMO_PATH = ROOT / "outputs" / "sample_research_memo.md"


st.set_page_config(
    page_title="Social Listening",
    page_icon="SL",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def prepare_articles(raw: pd.DataFrame) -> pd.DataFrame:
    classified = pd.DataFrame(classify_records(raw.to_dict(orient="records")))
    scored = add_scores(classified)
    return add_spike_scores(scored)


@st.cache_data
def load_sample_articles() -> pd.DataFrame:
    return prepare_articles(pd.read_csv(ARTICLE_PATH))


@st.cache_data
def load_operational_articles(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        generate_operational_demo_corpus(output_path=path)
    return prepare_articles(pd.read_csv(path))


@st.cache_data
def load_gdelt_articles(path: str) -> pd.DataFrame:
    return prepare_articles(pd.read_csv(path))


@st.cache_data
def load_bandit_log() -> pd.DataFrame:
    return pd.read_csv(BANDIT_LOG_PATH)


def filter_recent(df: pd.DataFrame, days_back: int) -> pd.DataFrame:
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    latest = data["date"].max()
    if pd.isna(latest):
        return data
    cutoff = latest - pd.Timedelta(days=days_back)
    return data[data["date"] >= cutoff].copy()


def data_quality_summary(df: pd.DataFrame) -> dict:
    dates = pd.to_datetime(df["date"], errors="coerce")
    geography_matches: list[str] = []
    if "geography_matches" in df:
        for value in df["geography_matches"].dropna():
            geography_matches.extend(part.strip() for part in str(value).split(";") if part.strip())
    top_geographies = (
        {str(key): int(value) for key, value in pd.Series(geography_matches).value_counts().head(5).items()}
        if geography_matches
        else {}
    )
    source_column = "domain" if "domain" in df else "source_name"
    top_sources = {
        str(key): int(value)
        for key, value in df[source_column].fillna(df["source_name"]).value_counts().head(5).items()
    }
    return {
        "articles": len(df),
        "date_range": f"{dates.min().date()} to {dates.max().date()}" if not dates.empty else "n/a",
        "top_geographies": top_geographies,
        "top_sources": top_sources,
    }


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] {
            font-family: 'Manrope', Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .stApp {
            background:
                radial-gradient(circle at 18% 0%, rgba(79, 70, 229, 0.08), transparent 28%),
                linear-gradient(180deg, #fbfbfc 0%, #f5f6f8 44%, #ffffff 100%);
            color: #15171c;
        }
        h1, h2, h3 {
            letter-spacing: 0;
            color: #111318;
        }
        .block-container {
            max-width: 1280px;
            padding-top: 2.1rem;
            padding-bottom: 4rem;
        }
        div[data-testid="stMetric"],
        div[data-testid="stDataFrame"],
        div[data-testid="stVegaLiteChart"] {
            border-radius: 18px;
        }
        .hero-panel {
            background: linear-gradient(135deg, #111318 0%, #202532 58%, #394150 100%);
            color: #f8fafc;
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 28px;
            padding: 34px 38px;
            margin: 18px 0 18px 0;
            box-shadow: 0 28px 70px rgba(17, 19, 24, 0.25);
            min-height: 250px;
        }
        .eyebrow {
            color: #b9c1d1;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-weight: 700;
            margin-bottom: 18px;
        }
        .hero-title {
            color: #ffffff;
            font-size: clamp(2rem, 4.2vw, 4.6rem);
            line-height: 0.98;
            font-weight: 800;
            letter-spacing: -0.035em;
            max-width: 980px;
            margin-bottom: 18px;
        }
        .hero-subtext {
            color: #d8dee9;
            font-size: 1.08rem;
            line-height: 1.65;
            max-width: 860px;
        }
        .insight-card {
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(17, 19, 24, 0.07);
            border-radius: 22px;
            padding: 21px 22px;
            min-height: 162px;
            box-shadow: 0 18px 50px rgba(17, 19, 24, 0.07);
            backdrop-filter: blur(18px);
        }
        .summary-label {
            color: #69707d;
            font-size: 0.78rem;
            font-weight: 750;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .summary-body {
            color: #14171f;
            font-size: 1.05rem;
            line-height: 1.55;
        }
        .principle-card {
            background: #ffffff;
            border: 1px solid rgba(17, 19, 24, 0.08);
            border-radius: 18px;
            padding: 20px;
            min-height: 184px;
            box-shadow: 0 14px 36px rgba(17, 19, 24, 0.06);
        }
        .radar-card {
            background: #ffffff;
            border: 1px solid rgba(17, 19, 24, 0.08);
            border-radius: 18px;
            padding: 20px;
            margin-bottom: 14px;
            box-shadow: 0 12px 28px rgba(17, 19, 24, 0.05);
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
        .definition-note {
            color: #555d6b;
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(17, 19, 24, 0.07);
            border-radius: 16px;
            padding: 16px 18px;
            line-height: 1.55;
            margin: 10px 0 18px 0;
        }
        div[data-testid="stDataFrame"] div {
            white-space: normal !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 10px 16px;
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


def strategic_context(df: pd.DataFrame, rollup: pd.DataFrame) -> dict:
    top_issue = rollup.sort_values(["spike_score", "mentions", "avg_intensity"], ascending=False).iloc[0]
    second_issue = rollup.sort_values(["mentions", "spike_score"], ascending=False).iloc[min(1, len(rollup) - 1)]
    geo_counts = geography_counts(df.to_dict(orient="records"))
    top_geos = list(geo_counts.keys())[:2] or ["Statewide"]
    top_geo_text = " and ".join(top_geos)
    spike = float(top_issue["spike_score"])
    spike_text = f"discussion volume is about {max(spike, 1):.1f}x above recent baseline"
    action = "move into message testing" if top_issue["radar_flag"] == "test" else "keep under analyst monitoring"
    return {
        "hero": f"{top_issue['classified_issue_area']} is moving fastest in {top_geo_text}.",
        "subtext": (
            f"{top_issue['classified_issue_area']} and {second_issue['classified_issue_area']} are now the leading "
            f"signals in the monitoring window. Operationally, {spike_text}, which means strategists should "
            f"{action} rather than wait for another news cycle."
        ),
        "concern": (
            f"Strongest emerging concern: {top_issue['classified_issue_area']}. "
            f"Discussion volume is about {max(spike, 1):.1f}x above recent baseline."
        ),
        "geography": f"Most discussion is concentrated in {top_geo_text}.",
        "action": f"Recommended action: {action} with human analyst review.",
    }


def render_header(df: pd.DataFrame, rollup: pd.DataFrame, data_label: str) -> None:
    context = strategic_context(df, rollup)
    st.title("Social Listening")
    st.caption("Narrative intelligence prototype for campaign research.")
    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="eyebrow">Adaptive intelligence briefing | {len(df):,} discourse artifacts | {data_label}</div>
            <div class="hero-title">{context['hero']}</div>
            <div class="hero-subtext">{context['subtext']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cards = [
        ("Emerging concern", context["concern"]),
        ("Geographic concentration", context["geography"]),
        ("Strategic next step", context["action"]),
    ]
    columns = st.columns([1.05, 1, 1])
    for column, (label, body) in zip(columns, cards):
        column.markdown(
            f"""
            <div class="insight-card">
                <div class="summary-label">{label}</div>
                <div class="summary-body">{body}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.write("")
    st.write("**Narrative Momentum**")
    momentum = daily_issue_volume(df)
    st.area_chart(momentum, x="date", y="mentions", color="classified_issue_area", height=360)


def render_overview(df: pd.DataFrame, rollup: pd.DataFrame) -> None:
    volume, issue_mix, tone = chart_dataframes(df)
    st.subheader("Overview")
    st.write("Strategic readout of issue movement, geography concentration, and research priority.")
    if "is_synthetic_operational_demo" in df.columns and bool(df["is_synthetic_operational_demo"].fillna(False).any()):
        st.info(
            "Synthetic operational-scale demo corpus generated from real NY discourse patterns. "
            "It simulates statewide monitoring volume while preserving observed issue and geography signals."
        )
    if "public news via GDELT" in set(df["source_type"].dropna()):
        summary = data_quality_summary(df)
        st.info(
            "Real-data quality note: "
            f"{summary['articles']} articles loaded | {summary['date_range']} | "
            f"top geographies: {summary['top_geographies']} | "
            f"top sources: {summary['top_sources']}"
        )
    st.divider()

    left, right = st.columns([1.2, 1])
    with left:
        st.line_chart(volume, x="date", y="mentions", color="classified_issue_area", height=340)
    with right:
        st.bar_chart(issue_mix, x="issue_area", y="mentions", height=340)

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
            height=280,
            column_config={
                "issue_area": st.column_config.TextColumn("issue area", width="medium"),
                "radar_flag": st.column_config.TextColumn("recommended priority", width="small"),
            },
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
        "Transparent rules convert public discourse into issue movement, tone, and research priority."
    )
    st.markdown(
        """
        <div class="definition-note">
        <strong>How to read this:</strong> spike score estimates how far discussion volume is above the recent
        baseline; a score near 3 means roughly 3x normal volume. Intensity combines keyword density, urgency
        language, and source amplification. These are triage signals for strategists, not truth labels.
        </div>
        """,
        unsafe_allow_html=True,
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
            Discussion volume is about {max(float(selected_row['spike_score']), 1):.1f}x recent baseline |
            Narrative intensity {selected_row['avg_intensity']:.1f} |
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
        height=420,
        column_config={
            "headline": st.column_config.TextColumn("headline", width="large"),
            "snippet": st.column_config.TextColumn("snippet", width="large"),
            "date": st.column_config.TextColumn("date", width="small"),
        },
    )


def render_memo(df: pd.DataFrame, rollup: pd.DataFrame) -> None:
    st.subheader("Research Memo")
    memo = generate_research_memo(df, rollup)
    MEMO_PATH.parent.mkdir(exist_ok=True)
    MEMO_PATH.write_text(memo, encoding="utf-8")
    st.markdown(memo)


def render_research_outputs(df: pd.DataFrame, rollup: pd.DataFrame) -> None:
    st.subheader("Research Outputs")
    st.write(
        "Human-readable campaign research artifacts generated from the current narrative signal set. "
        "These are the immediate strategist-facing outputs of the prototype."
    )
    outputs = export_research_outputs(df, rollup, ROOT / "outputs")
    paths = outputs["paths"]

    st.write("**Weekly Issue Brief**")
    st.dataframe(
        outputs["weekly_issue_brief_table"],
        hide_index=True,
        width="stretch",
        height=360,
        column_config={
            "interpretation": st.column_config.TextColumn("interpretation", width="large"),
            "top_sources": st.column_config.TextColumn("top sources", width="medium"),
            "top_geographies": st.column_config.TextColumn("top geographies", width="medium"),
        },
    )
    st.download_button(
        "Download weekly_issue_brief.md",
        data=Path(paths["weekly_issue_brief"]).read_text(encoding="utf-8"),
        file_name="weekly_issue_brief.md",
        mime="text/markdown",
    )

    st.write("**County / Geography Watchlist**")
    st.dataframe(
        outputs["geography_watchlist"],
        hide_index=True,
        width="stretch",
        height=420,
        column_config={
            "why_it_matters": st.column_config.TextColumn("why it matters", width="large"),
            "recommended_next_step": st.column_config.TextColumn("recommended next step", width="medium"),
        },
    )
    st.download_button(
        "Download geography_watchlist.csv",
        data=outputs["geography_watchlist"].to_csv(index=False),
        file_name="geography_watchlist.csv",
        mime="text/csv",
    )

    st.write("**Message Hypothesis Bank**")
    st.dataframe(
        outputs["message_hypothesis_bank"],
        hide_index=True,
        width="stretch",
        height=460,
        column_config={
            "economic_frame": st.column_config.TextColumn("economic frame", width="large"),
            "competence_frame": st.column_config.TextColumn("competence frame", width="large"),
            "values_trust_frame": st.column_config.TextColumn("values/trust frame", width="large"),
            "recommended_audience_hypothesis": st.column_config.TextColumn("audience hypothesis", width="large"),
            "risk_or_caveat": st.column_config.TextColumn("risk/caveat", width="large"),
        },
    )
    st.download_button(
        "Download message_hypothesis_bank.csv",
        data=outputs["message_hypothesis_bank"].to_csv(index=False),
        file_name="message_hypothesis_bank.csv",
        mime="text/csv",
    )

    st.write("**Polling / Focus Group Questions**")
    st.markdown(outputs["research_questions_md"])
    st.download_button(
        "Download research_questions.md",
        data=Path(paths["research_questions"]).read_text(encoding="utf-8"),
        file_name="research_questions.md",
        mime="text/markdown",
    )


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
        st.dataframe(context_features, hide_index=True, width="stretch", height=360)
    with right:
        st.write("**2. Candidate Message Arms**")
        st.caption("Each arm is a testable research hypothesis, not a persuasion claim.")
        st.dataframe(
            generate_message_arms(selected_issue),
            hide_index=True,
            width="stretch",
            height=250,
            column_config={"hypothesis": st.column_config.TextColumn("hypothesis", width="large")},
        )

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
        st.dataframe(bandit_log, hide_index=True, width="stretch", height=360)

    st.write("**4. Policy Simulator Output**")
    st.dataframe(
        simulate_policies(bandit_log),
        hide_index=True,
        width="stretch",
        height=220,
        column_config={"notes": st.column_config.TextColumn("notes", width="large")},
    )


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

    with st.sidebar:
        st.header("Demo Controls")
        st.caption("Use real public news by default, with sample data as a fallback.")
        data_source = st.radio(
            "Data source",
            options=["Operational-scale demo corpus", "Real GDELT data", "Sample demo data"],
            index=0,
        )
        st.caption(
            "Real public news volume from GDELT is relatively sparse for narrowly constrained NY political "
            "narratives, so the operational demo corpus extrapolates realistic statewide monitoring volume "
            "from observed patterns."
        )
        date_window = st.selectbox(
            "Date window",
            options=["Last 7 days", "Last 14 days", "Last 30 days"],
            index=1,
        )
        days_back = int(date_window.split()[1])

        gdelt_error = None
        if data_source == "Real GDELT data":
            if GDELT_PATH.exists():
                st.caption(f"Loaded `{GDELT_PATH.name}`. Fetch again for fresher public news.")
            else:
                st.info("No local GDELT cache yet.")
            if st.button("Fetch latest GDELT articles", type="primary"):
                try:
                    fetch_latest_gdelt_articles(days_back=days_back, output_path=GDELT_PATH)
                    load_gdelt_articles.clear()
                    st.success("Fetched latest GDELT articles.")
                    st.rerun()
                except Exception as exc:
                    gdelt_error = str(exc)
                    st.warning(f"GDELT fetch failed. Falling back to sample data. {gdelt_error}")

        if data_source == "Operational-scale demo corpus":
            df = filter_recent(load_operational_articles(str(OPERATIONAL_PATH)), days_back)
            data_label = "Operational demo corpus"
        elif data_source == "Real GDELT data" and GDELT_PATH.exists():
            try:
                df = filter_recent(load_gdelt_articles(str(GDELT_PATH)), days_back)
                data_label = "Latest GDELT article"
                if df.empty:
                    st.warning("No GDELT rows in the selected date window. Falling back to sample data.")
                    df = load_sample_articles()
                    data_label = "Latest sample"
            except Exception as exc:
                st.warning(f"Could not load GDELT cache. Falling back to sample data. {exc}")
                df = load_sample_articles()
                data_label = "Latest sample"
        else:
            if data_source == "Real GDELT data" and not GDELT_PATH.exists() and gdelt_error is None:
                st.warning("Real GDELT data has not been fetched yet. Showing sample data until you fetch.")
            df = load_sample_articles()
            data_label = "Latest sample"

        st.divider()
        st.caption("Filter the loaded corpus for walkthroughs.")
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
        st.caption("GDELT requires internet access, but no API key. Sample mode works offline.")

    rollup = issue_rollup(df)
    filtered = df[
        df["classified_issue_area"].isin(issues)
        & df["source_type"].isin(source_types)
    ].copy()
    filtered_rollup = issue_rollup(filtered) if not filtered.empty else rollup

    render_header(filtered, filtered_rollup, data_label)
    tab_overview, tab_radar, tab_memo, tab_outputs, tab_bandit, tab_what_is, tab_ethics = st.tabs(
        [
            "Overview",
            "Narrative Radar",
            "Research Memo",
            "Research Outputs",
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
    with tab_outputs:
        render_research_outputs(filtered, filtered_rollup)
    with tab_bandit:
        render_bandit_readiness(filtered, filtered_rollup)
    with tab_what_is:
        render_what_is()
    with tab_ethics:
        render_ethics()


if __name__ == "__main__":
    main()
