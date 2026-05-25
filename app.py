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
from src.classify_topics import classify_records
from src.collect_gdelt import fetch_latest_gdelt_articles
from src.collect_reddit import fetch_latest_reddit_posts
from src.generate_memo import generate_research_memo
from src.regions import NY_REGIONS, normalize_region_label
from src.research_outputs import export_research_outputs
from src.scoring import add_scores, add_spike_scores, daily_issue_volume, issue_rollup
from src.synthetic_corpus import generate_operational_demo_corpus

ROOT = Path(__file__).parent
ARTICLE_PATH = ROOT / "data" / "sample_articles.csv"
GDELT_PATH = ROOT / "data" / "gdelt_articles.csv"
REDDIT_PATH = ROOT / "data" / "reddit_posts.csv"
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
    classified["region"] = classified.apply(display_region, axis=1)
    classified["source_display"] = classified.apply(source_display_name, axis=1)
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
def load_reddit_posts(path: str) -> pd.DataFrame:
    return prepare_articles(pd.read_csv(path))


@st.cache_data
def load_combined_real_sources(gdelt_path: str | None, reddit_path: str | None) -> pd.DataFrame:
    frames = []
    if gdelt_path and Path(gdelt_path).exists():
        frames.append(pd.read_csv(gdelt_path))
    if reddit_path and Path(reddit_path).exists():
        frames.append(pd.read_csv(reddit_path))
    if not frames:
        raise FileNotFoundError("No real-source cache is available yet.")
    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw = raw.drop_duplicates(subset=["url"], keep="first").drop_duplicates(subset=["headline"], keep="first")
    return prepare_articles(raw)


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


def display_region(row: pd.Series) -> str:
    for column in ["region", "geography", "geography_matches", "detected_geographies", "geography_refs"]:
        if column in row and pd.notna(row[column]) and str(row[column]).strip():
            label = normalize_region_label(row[column])
            if label != "Statewide":
                return label
    text = f"{row.get('headline', '')} {row.get('snippet', '')}"
    return normalize_region_label(text)


def source_display_name(row: pd.Series) -> str:
    platform = str(row.get("source_platform", "")).lower()
    subreddit = str(row.get("subreddit", "")).strip()
    if platform == "reddit" or subreddit:
        return f"r/{subreddit}" if subreddit and not subreddit.startswith("r/") else subreddit or "Reddit"
    source_name = str(row.get("source_name", "")).strip()
    if source_name and source_name.lower() not in {"nan", "none"}:
        if "." in source_name and " " not in source_name:
            label = source_name.lower().replace("www.", "").split("/")[0].split(".")[0]
            return label.replace("-", " ").replace("_", " ").title()
        return source_name
    domain = str(row.get("domain", "")).strip().lower()
    if not domain or domain in {"nan", "none"}:
        return "Public source"
    domain = domain.replace("www.", "").split("/")[0]
    label = domain.split(".")[0].replace("-", " ").replace("_", " ").strip()
    return label.title() if label else domain


def clear_overview_filters() -> None:
    st.session_state["selected_overview_issue"] = None
    st.session_state["overview_regions"] = []
    st.session_state["overview_chart_reset"] = st.session_state.get("overview_chart_reset", 0) + 1


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
            grid-template-columns: 104px 1.1fr 0.9fr 112px minmax(320px, 2.5fr) 1.05fr;
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
        .story-tone {
            white-space: nowrap;
        }
        .story-title {
            font-weight: 650;
        }
        .empty-chart-note {
            color: #737985;
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(17, 19, 24, 0.07);
            border-radius: 16px;
            padding: 18px 20px;
            margin: 6px 0 18px 0;
        }
        .chart-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
            margin: 8px 0 18px 0;
            color: #68707d;
            font-size: 0.82rem;
        }
        .legend-pill {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            border-radius: 999px;
            padding: 7px 10px;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(17, 19, 24, 0.06);
            box-shadow: 0 8px 22px rgba(17, 19, 24, 0.035);
            white-space: nowrap;
        }
        .legend-dot {
            width: 9px;
            height: 9px;
            border-radius: 999px;
            display: inline-block;
        }
        .legend-line {
            width: 22px;
            border-radius: 999px;
            display: inline-block;
            background: #77808d;
        }
        .line-thick {
            height: 5px;
        }
        .line-thin {
            height: 2px;
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
        div[data-testid="stButton"] > button {
            border-radius: 999px;
            border: 1px solid rgba(17, 19, 24, 0.10);
            background: rgba(255, 255, 255, 0.78);
            box-shadow: 0 8px 24px rgba(17, 19, 24, 0.06);
            color: #20242c;
            font-weight: 650;
            min-height: 40px;
        }
        div[data-testid="stButton"] > button:hover {
            border-color: rgba(17, 19, 24, 0.18);
            background: rgba(255, 255, 255, 0.94);
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
            Social listening turns public news and public online discussion into a calmer research signal for
            campaign teams. This prototype uses extrapolated New York patterns to simulate a statewide discussion
            feed, while also supporting live GDELT news and public Reddit posts. It supports human strategists and
            future adaptive experimentation systems, but it is not voter microtargeting.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open dashboard", type="primary"):
        st.session_state["dashboard_opened"] = True
        st.rerun()


def display_geography(row: pd.Series) -> str:
    return display_region(row)


def filtered_for_overview(
    df: pd.DataFrame,
    selected_geos: list[str],
    selected_tones: list[str],
    selected_issue: str | None = None,
) -> pd.DataFrame:
    data = df.copy()
    data["display_region"] = data.apply(display_region, axis=1)
    if selected_issue:
        data = data[data["classified_issue_area"] == selected_issue]
    if selected_tones:
        data = data[data["tone"].isin(selected_tones)]
    if selected_geos:
        data = data[
            data["display_region"].apply(
                lambda value: any(region.strip() in selected_geos for region in str(value).split(","))
            )
        ]
    return data


def sentiment_label(score: float) -> str:
    if score <= -0.2:
        return "negative"
    if score >= 0.2:
        return "positive"
    return "mixed"


def issue_direction(issue_df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> str:
    daily = issue_daily_series(issue_df, start_date, end_date)
    if len(daily) < 4:
        return "Flat"
    midpoint = max(1, len(daily) // 2)
    early = float(daily.head(midpoint)["stories"].mean())
    recent = float(daily.tail(midpoint)["stories"].mean())
    if recent >= early * 1.12:
        return "Rising"
    if recent <= early * 0.88:
        return "Cooling"
    return "Flat"


def issue_daily_series(
    issue_df: pd.DataFrame,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    data = issue_df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize() if "date" in data else pd.NaT
    data = data.dropna(subset=["date"])
    if start_date is None:
        start_date = data["date"].min() if not data.empty else pd.NaT
    if end_date is None:
        end_date = data["date"].max() if not data.empty else pd.NaT
    if pd.isna(start_date) or pd.isna(end_date):
        return pd.DataFrame(columns=["date", "stories", "avg_sentiment"])
    date_index = pd.date_range(pd.to_datetime(start_date).normalize(), pd.to_datetime(end_date).normalize(), freq="D")
    if data.empty:
        daily = pd.DataFrame({"date": date_index, "stories": 0, "avg_sentiment": 0.0})
        return daily
    daily = (
        data.groupby("date")
        .agg(stories=("headline", "count"), avg_sentiment=("sentiment_score", "mean"))
        .reindex(date_index)
        .rename_axis("date")
        .reset_index()
    )
    daily["stories"] = daily["stories"].fillna(0).astype(int)
    daily["avg_sentiment"] = daily["avg_sentiment"].ffill().bfill().fillna(0)
    return daily


def issue_tone_color(issue_df: pd.DataFrame) -> str:
    score = float(issue_df["sentiment_score"].mean()) if not issue_df.empty else 0.0
    if score <= -0.10:
        return "#c75f5f"
    if score >= 0.10:
        return "#3d9a6a"
    return "#d8a923"


def issue_mini_chart(
    df: pd.DataFrame,
    issue: str,
    selected_issue: str | None = None,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> alt.Chart:
    issue_df = df[df["classified_issue_area"] == issue]
    chart_data = issue_daily_series(issue_df, start_date, end_date)
    chart_data["classified_issue_area"] = issue
    y_max = max(5, int(chart_data["stories"].max()) + 2)
    select_issue = alt.selection_point(name=f"issue_select_{issue_key(issue)}", fields=["classified_issue_area"], on="click", empty=False)
    hover = alt.selection_point(fields=["date"], nearest=True, on="mouseover", empty=False)

    base = alt.Chart(chart_data).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %d", labelAngle=-35, tickCount=6, grid=False)),
        y=alt.Y("stories:Q", title="Stories", scale=alt.Scale(domain=[0, y_max]), axis=alt.Axis(tickCount=3, grid=True)),
        color=alt.value(issue_tone_color(issue_df)),
        size=alt.Size("stories:Q", scale=alt.Scale(domain=[0, max(1, int(chart_data["stories"].max()))], range=[2.5, 7.5]), legend=None),
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("stories:Q", title="Stories"),
            alt.Tooltip("avg_sentiment:Q", title="Average tone", format=".2f"),
        ],
    )
    line = base.mark_trail().encode(
        opacity=alt.value(1 if selected_issue in {None, issue} else 0.28)
    )
    points = base.mark_circle(size=95).encode(opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    return (line + points).add_params(hover, select_issue).properties(height=142)


def issue_key(issue: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in issue.lower()).strip("_")


def issue_label(issue: str) -> str:
    if issue.startswith("AI"):
        return "AI" + issue[2:].capitalize()
    return issue.capitalize()


def selected_issue_from_mini_state(state: object) -> str | None:
    try:
        selection = dict(state.selection)  # type: ignore[attr-defined]
    except Exception:
        return None
    for raw in selection.values():
        if isinstance(raw, list) and raw:
            first = raw[0]
            if isinstance(first, dict) and first.get("classified_issue_area"):
                return str(first["classified_issue_area"])
        if isinstance(raw, dict):
            value = raw.get("classified_issue_area")
            if isinstance(value, list) and value:
                return str(value[0])
            if isinstance(value, str):
                return value
    return None


def render_story_table(df: pd.DataFrame) -> None:
    rows = df.sort_values("date", ascending=False).head(20)
    header = """
    <div class="story-row story-head">
        <div>Date</div><div>Issue</div><div>Region</div><div>Tone</div><div>Headline</div><div>Source</div>
    </div>
    """
    body = []
    for _, row in rows.iterrows():
        parsed_date = pd.to_datetime(row["date"], errors="coerce")
        date = f"{parsed_date.strftime('%b')} {parsed_date.day}, {parsed_date.year}" if not pd.isna(parsed_date) else ""
        issue = escape(str(row["classified_issue_area"]))
        geography = escape(display_region(row))
        tone = escape(str(row["tone"]))
        headline = escape(str(row["headline"]))
        source = escape(str(row.get("source_display", source_display_name(row))))
        body.append(
            f'<div class="story-row">'
            f'<div class="story-cell">{date}</div>'
            f'<div class="story-cell">{issue}</div>'
            f'<div class="story-cell">{geography}</div>'
            f'<div class="story-cell story-tone">{tone}</div>'
            f'<div class="story-cell story-title">{headline}</div>'
            f'<div class="story-cell">{source}</div>'
            f"</div>"
        )
    st.markdown(f"<div class='story-table'>{header}{''.join(body)}</div>", unsafe_allow_html=True)


def render_overview(df: pd.DataFrame, rollup: pd.DataFrame) -> None:
    render_onboarding()
    st.subheader("Discussion briefing")
    st.write("Track which topics are rising, where the conversation is moving, and which stories deserve research attention.")

    st.session_state.setdefault("overview_regions", [])
    filter_col, reset_col = st.columns([1.5, 0.45])
    selected_geos = filter_col.multiselect("Places", options=NY_REGIONS, key="overview_regions")
    reset_col.button("Clear filters", key="clear_overview_filters_button", on_click=clear_overview_filters)

    base = filtered_for_overview(df, selected_geos, [])
    if base.empty:
        st.warning("No stories match the selected filters.")
        return

    st.markdown("**Issue trends over time**")
    st.caption("Click an issue name to filter stories. Use Clear filters to return to the full view.")
    st.markdown(
        """
        <div class="chart-legend">
            <span class="legend-pill"><span class="legend-dot" style="background:#c75f5f"></span>Red = mostly negative coverage</span>
            <span class="legend-pill"><span class="legend-dot" style="background:#d8a923"></span>Yellow = mixed coverage</span>
            <span class="legend-pill"><span class="legend-dot" style="background:#3d9a6a"></span>Green = mostly constructive coverage</span>
            <span class="legend-pill"><span class="legend-line line-thick"></span>Thicker line = more stories/discussion items</span>
            <span class="legend-pill"><span class="legend-line line-thin"></span>Thinner line = fewer stories/discussion items</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    prior_issue = st.session_state.get("selected_overview_issue")
    if prior_issue and prior_issue not in set(base["classified_issue_area"]):
        prior_issue = None
        st.session_state["selected_overview_issue"] = None
    chart_dates = pd.to_datetime(base["date"], errors="coerce").dropna()
    chart_start = chart_dates.min().normalize()
    chart_end = chart_dates.max().normalize()

    for issue in sorted(df["classified_issue_area"].dropna().unique()):
        issue_df = base[base["classified_issue_area"] == issue]
        direction = issue_direction(issue_df, chart_start, chart_end) if not issue_df.empty else "Flat"
        title_col, direction_col = st.columns([4.4, 0.8])
        if title_col.button(issue_label(issue), key=f"issue_title_{issue_key(issue)}"):
            st.session_state["selected_overview_issue"] = None if prior_issue == issue else issue
            st.rerun()
        direction_col.markdown(f"<div class='small-muted'>{direction}</div>", unsafe_allow_html=True)
        if issue_df.empty:
            st.markdown(
                "<div class='empty-chart-note'>No stories found for this issue and place.</div>",
                unsafe_allow_html=True,
            )
            continue
        chart_key = f"issue_chart_{issue_key(issue)}_{st.session_state.get('overview_chart_reset', 0)}"
        chart_state = st.altair_chart(
            issue_mini_chart(base, issue, prior_issue, chart_start, chart_end),
            width="stretch",
            key=chart_key,
            on_select="rerun",
            selection_mode=[f"issue_select_{issue_key(issue)}"],
        )
        chart_issue = selected_issue_from_mini_state(chart_state)
        if chart_issue:
            st.session_state["selected_overview_issue"] = chart_issue
            prior_issue = chart_issue

    if prior_issue:
        st.caption(f"Showing stories for {issue_label(prior_issue)}. Line color reflects whether coverage is concerned, mixed, or constructive.")
    else:
        st.caption(
            "Change vs recent baseline compares current discussion volume with recent norms. "
            "Line color reflects average coverage tone: red is more concerned, yellow is mixed, green is more constructive."
        )

    current = filtered_for_overview(base, [], [], selected_issue=prior_issue)
    st.caption(
        "Coverage tone is a simple reading of public language: concerned, mixed, or constructive."
    )
    st.write("**Stories driving the movement**")
    render_story_table(current)


def render_memo(df: pd.DataFrame, rollup: pd.DataFrame) -> None:
    st.subheader("Research Memo")
    memo = generate_research_memo(df, rollup)
    MEMO_PATH.parent.mkdir(exist_ok=True)
    MEMO_PATH.write_text(memo, encoding="utf-8")
    st.markdown(memo)


def render_research_outputs(df: pd.DataFrame, rollup: pd.DataFrame) -> None:
    st.subheader("Research outputs")
    st.write(
        "Research directors, strategists, and message teams use these outputs to decide what deserves deeper "
        "investigation. The goal is to reduce noisy public discussion into a weekly brief, a regional watchlist, "
        "and concrete questions for polling, focus groups, and message testing."
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
        "This shows how discussion trends could feed future experimentation systems while preserving human review."
    )

    context_features = build_context_features(df, rollup)
    selected_issue = st.selectbox(
        "Example structured experiment input",
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

    structured_input = {
        "context": context_sample.to_dict(orient="records")[0],
        "candidate_messages": arms[["message_arm", "hypothesis"]].to_dict(orient="records"),
        "reward_signals": REWARD_SIGNALS,
        "human_review_required": True,
    }
    st.write("**Example structured experiment input**")
    st.json(structured_input)


def render_about() -> None:
    st.subheader("About")
    st.markdown(
        """
        Social Listening is a public-discourse research prototype for campaign strategy teams. It helps identify
        which topics are rising, where discussion is increasing, whether coverage is positive or negative, and
        which stories should become polling, focus group, or message-testing questions.

        **Methodology:** public stories and posts are classified with transparent keyword rules, scored for tone,
        grouped into six New York regions, and converted into human research outputs. The default simulated
        statewide discussion feed shows the workflow at monitoring scale. Live GDELT news and public Reddit posts
        can be fetched as reproducible public-source inputs.

        **Data sources:** GDELT public news, Reddit public posts, and a simulated statewide discussion feed.
        Real campaign systems would usually combine public news, public discussion, creator/media monitoring,
        polling, field notes, and campaign-owned engagement data.

        **Limitations:** public and aggregate-only analysis; no private voter data; no voter microtargeting;
        no attempt to identify individuals; no demographic profiling; no persuasion claims. Production use would
        require platform compliance, privacy/legal review, analyst review, and experimental safeguards.

        **David Rauch:** [LinkedIn](https://www.linkedin.com/in/davidwrauch/)  
        **GitHub repository:** [social-listening](https://github.com/davidwrauch/social-listening)
        """
    )


def main() -> None:
    inject_css()

    with st.sidebar:
        st.header("Controls")
        st.caption("Choose the discussion feed and time period.")
        data_source = st.radio(
            "Data source",
            options=[
                "Simulated statewide discussion feed",
                "Real public news and Reddit discussion",
                "Real GDELT public news only",
            ],
            index=0,
        )
        st.caption(
            "This prototype combines open public news signals with public online discussion signals. "
            "The simulated statewide feed demonstrates how the same workflow behaves at campaign-monitoring scale. "
            "GDELT and Reddit are included because they are reproducible public sources, not because they represent "
            "the full universe of voter opinion."
        )
        st.caption(
            "Real campaign systems would usually combine public news, public discussion, creator/media monitoring, "
            "polling, field notes, and campaign-owned engagement data."
        )
        date_window = st.selectbox(
            "Stories analyzed",
            options=["Last 7 days", "Last 14 days", "Last 30 days"],
            index=1,
        )
        days_back = int(date_window.split()[1])

        gdelt_error = None
        reddit_error = None
        if data_source in {"Real public news and Reddit discussion", "Real GDELT public news only"}:
            if not GDELT_PATH.exists():
                st.info("No local GDELT cache yet.")
            if st.button("Fetch latest GDELT news", type="primary"):
                try:
                    fetch_latest_gdelt_articles(days_back=days_back, output_path=GDELT_PATH)
                    load_gdelt_articles.clear()
                    load_combined_real_sources.clear()
                    st.success("Fetched latest GDELT articles.")
                    st.rerun()
                except Exception as exc:
                    gdelt_error = str(exc)
                    st.warning(f"GDELT fetch failed. The app can keep using the simulated statewide feed. {gdelt_error}")

        if data_source == "Real public news and Reddit discussion":
            if not REDDIT_PATH.exists():
                st.info("No local Reddit cache yet.")
            if st.button("Fetch latest Reddit posts"):
                try:
                    fetch_latest_reddit_posts(days_back=days_back, output_path=REDDIT_PATH)
                    load_reddit_posts.clear()
                    load_combined_real_sources.clear()
                    st.success("Fetched latest public Reddit posts.")
                    st.rerun()
                except Exception as exc:
                    reddit_error = str(exc)
                    st.warning(f"Reddit fetch failed or was rate-limited. The app can keep using the simulated statewide feed. {reddit_error}")

        if data_source == "Simulated statewide discussion feed":
            df = filter_recent(load_operational_articles(str(OPERATIONAL_PATH)), days_back)
        elif data_source == "Real public news and Reddit discussion" and (GDELT_PATH.exists() or REDDIT_PATH.exists()):
            try:
                df = filter_recent(
                    load_combined_real_sources(
                        str(GDELT_PATH) if GDELT_PATH.exists() else None,
                        str(REDDIT_PATH) if REDDIT_PATH.exists() else None,
                    ),
                    days_back,
                )
                if df.empty:
                    st.warning("No real-source rows in the selected time period. Showing the simulated statewide feed.")
                    df = filter_recent(load_operational_articles(str(OPERATIONAL_PATH)), days_back)
            except Exception as exc:
                st.warning(f"Could not load real-source caches. Showing the simulated statewide feed. {exc}")
                df = filter_recent(load_operational_articles(str(OPERATIONAL_PATH)), days_back)
        elif data_source == "Real GDELT public news only" and GDELT_PATH.exists():
            try:
                df = filter_recent(load_gdelt_articles(str(GDELT_PATH)), days_back)
                if df.empty:
                    st.warning("No GDELT rows in the selected time period. Showing the simulated statewide feed.")
                    df = filter_recent(load_operational_articles(str(OPERATIONAL_PATH)), days_back)
            except Exception as exc:
                st.warning(f"Could not load GDELT cache. Showing the simulated statewide feed. {exc}")
                df = filter_recent(load_operational_articles(str(OPERATIONAL_PATH)), days_back)
        else:
            if data_source != "Simulated statewide discussion feed" and gdelt_error is None and reddit_error is None:
                st.warning("Real public-source data has not been fetched yet. Showing the simulated statewide feed.")
            df = filter_recent(load_operational_articles(str(OPERATIONAL_PATH)), days_back)

        st.divider()
        st.caption("GDELT and Reddit require internet access, but no API key for this basic demo.")

    rollup = issue_rollup(df)
    filtered = df.copy()
    filtered_rollup = rollup

    render_app_title()
    tab_overview, tab_memo, tab_outputs, tab_experimentation, tab_about = st.tabs(
        [
            "Overview",
            "Research memo",
            "Research outputs",
            "For experimentation",
            "About",
        ]
    )
    with tab_overview:
        render_overview(filtered, filtered_rollup)
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
