from __future__ import annotations

from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.bandit_simulator import (
    REWARD_SIGNALS,
    build_context_features,
    generate_message_arms,
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
            max-width: 1320px;
            padding-top: 1.6rem;
            padding-bottom: 4rem;
        }
        div[data-testid="stMetric"],
        div[data-testid="stDataFrame"],
        div[data-testid="stVegaLiteChart"] {
            border-radius: 18px;
        }
        .briefing-banner {
            background: linear-gradient(135deg, #111318 0%, #202532 62%, #343d4c 100%);
            color: #f8fafc;
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 28px;
            padding: 30px 34px;
            margin: 12px 0 26px 0;
            box-shadow: 0 26px 70px rgba(17, 19, 24, 0.22);
        }
        .eyebrow {
            color: #b9c1d1;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-weight: 700;
            margin-bottom: 18px;
        }
        .briefing-title {
            color: #ffffff;
            font-size: clamp(1.9rem, 3.3vw, 3.5rem);
            line-height: 1.03;
            font-weight: 800;
            letter-spacing: -0.035em;
            max-width: 1060px;
            margin-bottom: 18px;
        }
        .briefing-copy {
            color: #d8dee9;
            font-size: 1.08rem;
            line-height: 1.65;
            max-width: 940px;
        }
        .soft-panel {
            background: rgba(255,255,255,0.80);
            border: 1px solid rgba(17, 19, 24, 0.07);
            border-radius: 22px;
            padding: 22px;
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
        .story-table {
            display: grid;
            gap: 10px;
            margin-top: 14px;
        }
        .story-row {
            display: grid;
            grid-template-columns: 92px 1.15fr 0.9fr 0.8fr minmax(280px, 2.4fr) 1fr;
            gap: 14px;
            align-items: start;
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(17,19,24,0.07);
            border-radius: 18px;
            padding: 15px 17px;
            box-shadow: 0 10px 28px rgba(17,19,24,0.045);
        }
        .story-head {
            color: #69707d;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 800;
            box-shadow: none;
            background: transparent;
            border: 0;
            padding-bottom: 0;
        }
        .story-cell {
            color: #1d222b;
            font-size: 0.92rem;
            line-height: 1.45;
            overflow-wrap: anywhere;
        }
        .story-title {
            font-weight: 650;
        }
        @media (max-width: 900px) {
            .story-row {
                grid-template-columns: 1fr;
            }
            .story-head {
                display: none;
            }
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


def render_app_title() -> None:
    st.title("Social Listening")
    st.caption("Narrative intelligence prototype for campaign research.")


def render_onboarding() -> None:
    if st.session_state.get("dashboard_opened", False):
        return
    st.markdown(
        """
        <div class="briefing-banner">
            <div class="eyebrow">Briefing system for public discourse</div>
            <div class="briefing-title">See which stories are rising, where they are moving, and what researchers should investigate next.</div>
            <div class="briefing-copy">
            Social listening turns public news and discussion into a calmer research signal for campaign teams.
            This prototype uses extrapolated GDELT-derived New York patterns to simulate a statewide discussion feed,
            while still allowing live public-news updates. It supports human strategists and future adaptive
            experimentation systems, but it is not voter microtargeting.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open dashboard", type="primary"):
        st.session_state["dashboard_opened"] = True
        st.rerun()


def display_geography(row: pd.Series) -> str:
    for column in ["geography_matches", "detected_geographies", "geography_refs"]:
        if column in row and pd.notna(row[column]) and str(row[column]).strip():
            return str(row[column]).replace(";", ", ")
    return "Statewide"


def filtered_for_overview(df: pd.DataFrame, selected_issues: list[str], selected_geos: list[str], selected_tones: list[str]) -> pd.DataFrame:
    data = df.copy()
    data["display_geography"] = data.apply(display_geography, axis=1)
    if selected_issues:
        data = data[data["classified_issue_area"].isin(selected_issues)]
    if selected_tones:
        data = data[data["tone"].isin(selected_tones)]
    if selected_geos:
        pattern = "|".join(selected_geos)
        data = data[data["display_geography"].str.contains(pattern, case=False, regex=True, na=False)]
    return data


def sentiment_label(score: float) -> str:
    if score <= -0.2:
        return "negative"
    if score >= 0.2:
        return "positive"
    return "mixed"


def overview_chart(df: pd.DataFrame, selected_issues: list[str]) -> alt.Chart:
    chart_data = (
        df.assign(date=pd.to_datetime(df["date"]).dt.date)
        .groupby(["date", "classified_issue_area"])
        .agg(stories=("headline", "count"), avg_sentiment=("sentiment_score", "mean"))
        .reset_index()
    )
    chart_data["coverage"] = chart_data["avg_sentiment"].apply(sentiment_label)
    hover = alt.selection_point(fields=["date"], nearest=True, on="mouseover", empty=False)

    if len(selected_issues) == 1:
        color = alt.Color(
            "coverage:N",
            scale=alt.Scale(domain=["negative", "mixed", "positive"], range=["#d65f5f", "#d5a11e", "#2e9d68"]),
            legend=alt.Legend(title="Coverage tone"),
        )
    else:
        legend_select = alt.selection_point(fields=["classified_issue_area"], bind="legend")
        color = alt.Color("classified_issue_area:N", legend=alt.Legend(title="Issue"))

    base = alt.Chart(chart_data).encode(
        x=alt.X("date:T", title="Time"),
        y=alt.Y("stories:Q", title="Stories / discussion items"),
        color=color,
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("classified_issue_area:N", title="Issue"),
            alt.Tooltip("stories:Q", title="Stories"),
            alt.Tooltip("coverage:N", title="Coverage tone"),
        ],
    )
    line = base.mark_line(point=True, strokeWidth=3).encode(
        opacity=alt.condition(legend_select, alt.value(1), alt.value(0.16)) if len(selected_issues) != 1 else alt.value(1)
    )
    points = base.mark_circle(size=95).encode(opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    if len(selected_issues) != 1:
        line = line.add_params(legend_select)
    return (line + points.add_params(hover)).properties(height=430)


def render_story_table(df: pd.DataFrame) -> None:
    rows = df.sort_values("date", ascending=False).head(20)
    header = """
    <div class="story-row story-head">
        <div>Date</div><div>Issue</div><div>Geography</div><div>Sentiment</div><div>Headline</div><div>Source</div>
    </div>
    """
    body = []
    source_col = "domain" if "domain" in rows.columns else "source_name"
    for _, row in rows.iterrows():
        date = pd.to_datetime(row["date"]).strftime("%b %d")
        issue = escape(str(row["classified_issue_area"]))
        geography = escape(display_geography(row))
        tone = escape(str(row["tone"]))
        headline = escape(str(row["headline"]))
        source = escape(str(row.get(source_col, row.get("source_name", ""))))
        body.append(
            f'<div class="story-row">'
            f'<div class="story-cell">{date}</div>'
            f'<div class="story-cell">{issue}</div>'
            f'<div class="story-cell">{geography}</div>'
            f'<div class="story-cell">{tone}</div>'
            f'<div class="story-cell story-title">{headline}</div>'
            f'<div class="story-cell">{source}</div>'
            f"</div>"
        )
    st.markdown(f"<div class='story-table'>{header}{''.join(body)}</div>", unsafe_allow_html=True)


def render_overview(df: pd.DataFrame, rollup: pd.DataFrame) -> None:
    render_onboarding()
    st.subheader("Discussion briefing")
    st.write("Track which topics are rising, where the conversation is moving, and which stories deserve research attention.")

    all_geos = sorted({geo.strip() for value in df.apply(display_geography, axis=1) for geo in str(value).split(",") if geo.strip()})
    controls = st.columns([1.1, 1, 0.9])
    selected_issues = controls[0].multiselect(
        "Topics",
        options=sorted(df["classified_issue_area"].unique()),
        default=sorted(df["classified_issue_area"].unique()),
    )
    selected_geos = controls[1].multiselect("Places", options=all_geos, default=[])
    selected_tones = controls[2].multiselect("Coverage tone", options=sorted(df["tone"].unique()), default=[])

    current = filtered_for_overview(df, selected_issues, selected_geos, selected_tones)
    if current.empty:
        st.warning("No stories match the selected filters.")
        return

    st.altair_chart(overview_chart(current, selected_issues), width="stretch")
    st.caption(
        "Change vs recent baseline compares current discussion volume with recent norms. "
        "When one topic is selected, line color reflects whether coverage is negative, mixed, or positive."
    )
    st.write("**Stories driving the movement**")
    render_story_table(current)


def render_narrative_radar(df: pd.DataFrame, rollup: pd.DataFrame) -> None:
    st.subheader("Narrative radar")
    st.write(
        "Transparent rules convert public discourse into issue movement, tone, and research priority."
    )
    st.markdown(
        """
        <div class="definition-note">
        <strong>How to read this:</strong> change vs recent baseline estimates how far discussion volume is above
        normal. A value near 3 means roughly 3x normal volume. The attention-and-tone signal combines repeated
        keywords, urgency language, and source amplification. These are research triage signals, not truth labels.
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
            Attention and tone {selected_row['avg_intensity']:.1f} |
            Average sentiment {selected_row['avg_sentiment']:.2f}
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
            "narrative_intensity": st.column_config.NumberColumn("attention and tone"),
            "spike_score": st.column_config.NumberColumn("change vs baseline"),
            "keyword_hits": st.column_config.TextColumn("matching terms", width="medium"),
        },
    )


def render_memo(df: pd.DataFrame, rollup: pd.DataFrame) -> None:
    st.subheader("Research Memo")
    memo = generate_research_memo(df, rollup)
    MEMO_PATH.parent.mkdir(exist_ok=True)
    MEMO_PATH.write_text(memo, encoding="utf-8")
    st.markdown(memo)


def render_research_outputs(df: pd.DataFrame, rollup: pd.DataFrame) -> None:
    st.subheader("Research outputs")
    st.write(
        "These outputs are for research directors, strategists, and message teams. They reduce noisy public "
        "discussion into briefs, geography watchlists, message hypotheses, and questions for polling, focus "
        "groups, or message testing."
    )
    outputs = export_research_outputs(df, rollup, ROOT / "outputs")
    paths = outputs["paths"]

    st.write("**Weekly issue brief**")
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

    st.write("**County / geography watchlist**")
    st.dataframe(
        outputs["geography_watchlist"],
        hide_index=True,
        width="stretch",
        height=420,
        column_config={
            "why_it_matters": st.column_config.TextColumn("why it matters", width="large"),
            "recommended_next_step": st.column_config.TextColumn("recommended next step", width="medium"),
            "change_vs_recent_baseline": st.column_config.NumberColumn("change vs recent baseline"),
        },
    )
    st.download_button(
        "Download geography_watchlist.csv",
        data=outputs["geography_watchlist"].to_csv(index=False),
        file_name="geography_watchlist.csv",
        mime="text/csv",
    )

    st.write("**Message hypothesis bank**")
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

    st.write("**Polling / focus group questions**")
    st.markdown(outputs["research_questions_md"])
    st.download_button(
        "Download research_questions.md",
        data=Path(paths["research_questions"]).read_text(encoding="utf-8"),
        file_name="research_questions.md",
        mime="text/markdown",
    )


def render_for_experimentation(df: pd.DataFrame, rollup: pd.DataFrame) -> None:
    st.subheader("For experimentation")
    st.write(
        "This is how narrative intelligence could feed adaptive experimentation systems while preserving "
        "human review. The goal is to pass structured research context into future tests, not to optimize voters."
    )

    context_features = build_context_features(df, rollup)
    selected_issue = st.selectbox(
        "Issue for example payload",
        options=sorted(df["classified_issue_area"].unique()),
        key="bandit_issue",
    )
    arms = generate_message_arms(selected_issue)
    context_sample = context_features[context_features["issue_area"] == selected_issue].head(1)
    if context_sample.empty:
        context_sample = context_features.head(1)

    left, right = st.columns([1, 1])
    with left:
        st.write("**Context features**")
        st.caption("These are issue and discourse features, not private voter records.")
        st.dataframe(context_features, hide_index=True, width="stretch", height=360)
    with right:
        st.write("**Message hypotheses**")
        st.caption("Each row is a direction a strategist could review before any test.")
        st.dataframe(
            arms,
            hide_index=True,
            width="stretch",
            height=280,
            column_config={"hypothesis": st.column_config.TextColumn("hypothesis", width="large")},
        )

    st.write("**Reward definitions**")
    reward_cols = st.columns(3)
    for idx, signal in enumerate(REWARD_SIGNALS):
        reward_cols[idx % 3].markdown(f"- {signal}")
    st.caption(
        "Rewards are research and engagement signals. This demo does not claim measured persuasion effects."
    )

    payload = {
        "context": context_sample.to_dict(orient="records")[0],
        "candidate_messages": arms[["message_arm", "hypothesis"]].to_dict(orient="records"),
        "reward_signals": REWARD_SIGNALS,
        "human_review_required": True,
    }
    st.write("**Example experiment payload**")
    st.json(payload)


def render_about() -> None:
    st.subheader("About")
    st.markdown(
        """
        Social Listening is a public-discourse research prototype for campaign strategy teams. It helps identify
        which topics are rising, where discussion is increasing, whether coverage is positive or negative, and
        which stories should become polling, focus group, or message-testing questions.

        Methodology is intentionally transparent: public stories are classified with keyword rules, scored for
        tone, grouped by issue and place, and converted into human research outputs. The default simulated
        statewide discussion feed is extrapolated from GDELT-derived New York patterns because narrowly
        constrained live public-news volume can be sparse.

        Limitations: no private voter data, no voter microtargeting, no demographic modeling, and no claim of
        measured persuasion effects. Production use would require platform compliance, privacy/legal review,
        human analyst review, and experimental safeguards.

        [LinkedIn](https://www.linkedin.com/in/davidwrauch/)  
        [GitHub](https://github.com/davidwrauch/social-listening)
        """
    )


def main() -> None:
    inject_css()

    with st.sidebar:
        st.header("Controls")
        st.caption("Choose the discussion feed and time period.")
        data_source = st.radio(
            "Data source",
            options=["Simulated statewide discussion feed", "Real public news (GDELT)", "Small sample feed"],
            index=0,
        )
        st.caption(
            "Real public news volume from GDELT is relatively sparse for narrowly constrained NY political "
            "narratives, so the simulated statewide feed extrapolates realistic monitoring volume "
            "from observed patterns."
        )
        date_window = st.selectbox(
            "Stories analyzed",
            options=["Last 7 days", "Last 14 days", "Last 30 days"],
            index=1,
        )
        days_back = int(date_window.split()[1])

        gdelt_error = None
        if data_source == "Real public news (GDELT)":
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

        if data_source == "Simulated statewide discussion feed":
            df = filter_recent(load_operational_articles(str(OPERATIONAL_PATH)), days_back)
        elif data_source == "Real public news (GDELT)" and GDELT_PATH.exists():
            try:
                df = filter_recent(load_gdelt_articles(str(GDELT_PATH)), days_back)
                if df.empty:
                    st.warning("No GDELT rows in the selected time period. Falling back to sample data.")
                    df = load_sample_articles()
            except Exception as exc:
                st.warning(f"Could not load GDELT cache. Falling back to sample data. {exc}")
                df = load_sample_articles()
        else:
            if data_source == "Real public news (GDELT)" and not GDELT_PATH.exists() and gdelt_error is None:
                st.warning("Real GDELT data has not been fetched yet. Showing sample data until you fetch.")
            df = load_sample_articles()

        st.divider()
        st.caption("Filters")
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

    render_app_title()
    tab_overview, tab_radar, tab_memo, tab_outputs, tab_experimentation, tab_about = st.tabs(
        [
            "Overview",
            "Narrative radar",
            "Research memo",
            "Research outputs",
            "For experimentation",
            "About",
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
    with tab_experimentation:
        render_for_experimentation(filtered, filtered_rollup)
    with tab_about:
        render_about()


if __name__ == "__main__":
    main()
