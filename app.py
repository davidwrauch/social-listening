from __future__ import annotations

from html import escape
from pathlib import Path
import random

import altair as alt
import pandas as pd
import streamlit as st

from src.bandit_simulator import (
    REWARD_SIGNALS,
    build_context_features,
    generate_message_arms,
)
from src.classify_topics import classify_records
from src.generate_memo import generate_research_memo
from src.regions import NY_REGIONS, normalize_region_label
from src.research_outputs import export_research_outputs
from src.scoring import add_scores, add_spike_scores, daily_issue_volume, issue_rollup

ROOT = Path(__file__).parent
ARTICLE_PATH = ROOT / "data" / "sample_articles.csv"
GDELT_PATH = ROOT / "data" / "gdelt_articles.csv"
REDDIT_PATH = ROOT / "data" / "reddit_posts.csv"
BANDIT_LOG_PATH = ROOT / "data" / "sample_bandit_log.csv"
MEMO_PATH = ROOT / "outputs" / "sample_research_memo.md"


st.set_page_config(
    page_title="Social Listening",
    page_icon="SL",
    layout="wide",
    initial_sidebar_state="collapsed",
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
def load_monitoring_feed(gdelt_path: str | None, reddit_path: str | None, days_back: int = 14) -> pd.DataFrame:
    observed = load_combined_real_sources(gdelt_path, reddit_path)
    recent = filter_recent(observed, days_back)
    extended = extend_public_discourse_history(recent, days_back=days_back)
    return prepare_articles(extended)


@st.cache_data
def load_bandit_log() -> pd.DataFrame:
    return pd.read_csv(BANDIT_LOG_PATH)


def filter_recent(df: pd.DataFrame, days_back: int) -> pd.DataFrame:
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    latest = data["date"].max()
    if pd.isna(latest):
        return data
    cutoff = latest.normalize() - pd.Timedelta(days=max(days_back - 1, 0))
    return data[data["date"] >= cutoff].copy()


def extend_public_discourse_history(df: pd.DataFrame, days_back: int = 14) -> pd.DataFrame:
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"])
    if data.empty:
        return data

    latest = data["date"].max().normalize()
    date_index = pd.date_range(latest - pd.Timedelta(days=days_back - 1), latest, freq="D")
    issues = [
        "AI / tech jobs",
        "affordability / cost of living",
        "corruption / competence / trust",
        "housing / rent",
        "immigration / public safety",
    ]
    issue_distribution = data["classified_issue_area"].value_counts(normalize=True).to_dict()
    source_distribution = data["source_display"].value_counts(normalize=True).to_dict() if "source_display" in data else {}
    region_distribution = data["region"].value_counts(normalize=True).to_dict() if "region" in data else {}

    rows = []
    rng = random.Random(20260525)
    target_daily = _target_daily_volume(data, date_index)
    for day_position, day in enumerate(date_index):
        day_rows = data[data["date"].dt.normalize() == day]
        day_count = len(day_rows)
        target = target_daily[day_position]
        for issue in issues:
            observed_issue_count = len(day_rows[day_rows["classified_issue_area"] == issue])
            issue_weight = issue_distribution.get(issue, 0.12)
            issue_target = max(3, round(target * issue_weight * rng.uniform(0.78, 1.24)))
            gap = max(0, issue_target - observed_issue_count)
            for _ in range(gap):
                rows.append(_grounded_extension_row(data, issue, day, rng, source_distribution, region_distribution))

    if not rows:
        return data
    extension = pd.DataFrame(rows)
    return pd.concat([data, extension], ignore_index=True, sort=False)


def _target_daily_volume(data: pd.DataFrame, date_index: pd.DatetimeIndex) -> list[int]:
    observed = data.assign(day=data["date"].dt.normalize()).groupby("day").size()
    median_volume = int(max(45, min(95, observed[observed > 0].median() if not observed.empty else 60)))
    center = (len(date_index) - 1) / 2
    targets = []
    for idx, _ in enumerate(date_index):
        trend = 1 + 0.018 * idx
        wave = 1 + 0.18 * (1 - abs(idx - center) / max(center, 1))
        targets.append(max(38, round(median_volume * trend * wave)))
    return targets


def _grounded_extension_row(
    data: pd.DataFrame,
    issue: str,
    day: pd.Timestamp,
    rng: random.Random,
    source_distribution: dict[str, float],
    region_distribution: dict[str, float],
) -> dict:
    seed_pool = data[data["classified_issue_area"] == issue]
    if seed_pool.empty:
        seed_pool = data
    seed = seed_pool.sample(n=1, random_state=rng.randrange(1_000_000)).iloc[0]
    region = _weighted_choice(region_distribution, rng) or str(seed.get("region", "NYC"))
    source = _weighted_choice(source_distribution, rng) or str(seed.get("source_display", seed.get("source_name", "Public monitoring feed")))
    locality = _locality_for_region(region, rng)
    angle = _issue_angle(issue, rng)
    movement = rng.choice(["attention", "discussion", "concern", "questions", "coverage", "debate"])
    tone_word = rng.choice(["grows", "continues", "builds", "moves", "spreads"])
    minute = rng.randrange(0, 60)
    hour = rng.randrange(7, 22)
    timestamp = day + pd.Timedelta(hours=hour, minutes=minute)
    headline = f"{locality} {angle} as {movement} {tone_word}"
    snippet = (
        f"Grounded demo extension based on observed GDELT and Reddit patterns: {issue} discussion in "
        f"{region} is represented with natural variation for the 14-day monitoring view."
    )
    return {
        "date": day.date().isoformat(),
        "timestamp": timestamp.isoformat(),
        "source_name": "Grounded demo extension",
        "source_display": "Grounded demo extension",
        "source_type": "grounded demonstration extension",
        "source_platform": "grounded demo extension",
        "headline": headline,
        "snippet": snippet,
        "url": "",
        "issue_area": issue,
        "classified_issue_area": issue,
        "geography": region,
        "geography_refs": f"{locality}; {region}",
        "geography_matches": f"{locality}; {region}",
        "region": region,
        "domain": "",
        "language": "English",
        "is_grounded_demo_extension": True,
        "grounding_source": source,
    }


def _weighted_choice(distribution: dict[str, float], rng: random.Random) -> str | None:
    choices = [(str(key), float(value)) for key, value in distribution.items() if str(key).strip() and str(key) != "nan"]
    if not choices:
        return None
    total = sum(weight for _, weight in choices)
    pick = rng.random() * total
    running = 0.0
    for value, weight in choices:
        running += weight
        if running >= pick:
            return value
    return choices[-1][0]


def _locality_for_region(region: str, rng: random.Random) -> str:
    localities = {
        "NYC": ["Queens", "Brooklyn", "Bronx", "Manhattan", "Staten Island"],
        "Long Island": ["Nassau", "Suffolk", "Hempstead", "Huntington"],
        "Hudson Valley": ["Westchester", "Rockland", "Yonkers", "Orange County"],
        "Capital Region": ["Albany", "Schenectady", "Troy", "Saratoga"],
        "Central NY": ["Syracuse", "Utica", "Rome"],
        "Western NY": ["Buffalo", "Rochester", "Erie County", "Monroe County"],
    }
    return rng.choice(localities.get(region, ["New York"]))


def _issue_angle(issue: str, rng: random.Random) -> str:
    angles = {
        "affordability / cost of living": [
            "families describe grocery and utility pressure",
            "commuters connect fare pressure to household budgets",
            "residents ask for clearer cost-of-living plans",
        ],
        "housing / rent": [
            "renters describe instability and repair delays",
            "housing advocates warn that eviction pressure is rising",
            "tenants connect rent increases to neighborhood displacement",
        ],
        "immigration / public safety": [
            "residents link shelter planning to neighborhood quality of life",
            "officials debate public safety coordination and fairness",
            "community groups ask for clearer asylum support plans",
        ],
        "AI / tech jobs": [
            "workers ask whether AI investment will produce durable jobs",
            "regional employers highlight training needs around automation",
            "labor groups call for guardrails around artificial intelligence",
        ],
        "corruption / competence / trust": [
            "watchdogs press for accountability and procurement transparency",
            "voters connect ethics stories to broader competence concerns",
            "local coverage questions whether government can deliver reliably",
        ],
    }
    return rng.choice(angles.get(issue, ["residents discuss local campaign concerns"]))


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


def render_floating_clear_filters() -> None:
    st.markdown('<span class="clear-filter-anchor"></span>', unsafe_allow_html=True)
    st.button("Clear filters", key="floating_clear_filters_button", on_click=clear_overview_filters)


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
            flex-wrap: nowrap;
            gap: 8px;
            align-items: center;
            margin: 8px 0 18px 0;
            color: #68707d;
            font-size: 0.78rem;
            overflow-x: auto;
            padding-bottom: 2px;
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
        div[data-testid="stElementContainer"]:has(.clear-filter-anchor) {
            display: none;
        }
        .st-key-floating_clear_filters_button {
            position: fixed;
            right: 28px;
            top: 50%;
            transform: translateY(-50%);
            z-index: 9999;
        }
        .st-key-floating_clear_filters_button div[data-testid="stButton"] button,
        .st-key-floating_clear_filters_button button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 58px !important;
            height: 58px !important;
            padding: 0 28px !important;
            border-radius: 999px !important;
            background: rgba(255, 255, 255, 0.88) !important;
            border: 1px solid rgba(17, 19, 24, 0.10) !important;
            box-shadow: 0 18px 42px rgba(17, 19, 24, 0.14) !important;
            color: #20242c !important;
            font-size: 1.04rem !important;
            font-weight: 800 !important;
            backdrop-filter: blur(16px) !important;
        }
        .st-key-floating_clear_filters_button div[data-testid="stButton"] button:hover,
        .st-key-floating_clear_filters_button button:hover {
            background: rgba(255, 255, 255, 0.98) !important;
            border-color: rgba(17, 19, 24, 0.18) !important;
        }
        .story-source-link {
            color: #3454d1 !important;
            text-decoration: none !important;
            font-weight: 650;
        }
        .story-source-link:hover {
            text-decoration: underline !important;
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
    st.caption("A public-discourse intelligence system for campaign research.")
    st.write(
        "This prototype turns public news and Reddit discussion into a strategist-ready view of which issues are "
        "gaining attention across New York, where those conversations are moving, and which stories deserve "
        "research follow-up. It is designed for human campaign teams first: issue briefs, story evidence, "
        "message hypotheses, and polling questions."
    )
    st.write(
        "The same structured signals can also feed adaptive experimentation later, helping teams test message "
        "frames as new stories take up more of the public conversation. A production version could add polling, "
        "TV transcripts, creator monitoring, campaign responses, field notes, and campaign-owned engagement data."
    )
    st.caption(
        "Live public sources can be sparse for narrowly filtered New York political narratives, so this demo uses "
        "GDELT and Reddit as grounding signals and extends them into a realistic 14-day monitoring feed for demonstration."
    )


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


def issue_tone_color(issue_df: pd.DataFrame, tone_reference: pd.Series | None = None) -> str:
    score = float(issue_df["sentiment_score"].mean()) if not issue_df.empty else 0.0
    if tone_reference is not None and tone_reference.dropna().nunique() >= 3:
        low = float(tone_reference.quantile(0.34))
        high = float(tone_reference.quantile(0.67))
        if score <= low:
            return "#c75f5f"
        if score >= high:
            return "#3d9a6a"
        return "#d8a923"
    if score <= -0.06:
        return "#c75f5f"
    if score >= 0.06:
        return "#3d9a6a"
    return "#d8a923"


def issue_mini_chart(
    df: pd.DataFrame,
    issue: str,
    selected_issue: str | None = None,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    tone_reference: pd.Series | None = None,
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
        color=alt.value(issue_tone_color(issue_df, tone_reference)),
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


ISSUE_AREA_COLORS = {
    "AI / tech jobs": "#8fb5ff",
    "affordability / cost of living": "#f2b872",
    "corruption / competence / trust": "#b9a7e8",
    "housing / rent": "#7bc9b4",
    "immigration / public safety": "#f08f8f",
}

ISSUE_LEGEND_LABELS = {
    "AI / tech jobs": "AI/jobs",
    "affordability / cost of living": "Affordability",
    "corruption / competence / trust": "Trust/competence",
    "housing / rent": "Housing/rent",
    "immigration / public safety": "Immigration/safety",
}


def topic_share_chart(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> alt.Chart:
    issues = sorted(df["classified_issue_area"].dropna().unique())
    date_index = pd.date_range(start_date, end_date, freq="D")
    grid = pd.MultiIndex.from_product([date_index, issues], names=["date", "classified_issue_area"]).to_frame(index=False)
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    counts = data.groupby(["date", "classified_issue_area"]).size().reset_index(name="stories")
    chart_data = grid.merge(counts, on=["date", "classified_issue_area"], how="left")
    chart_data["stories"] = chart_data["stories"].fillna(0).astype(int)
    chart_data["legend_issue"] = chart_data["classified_issue_area"].map(ISSUE_LEGEND_LABELS)
    daily_totals = chart_data.groupby("date")["stories"].transform("sum")
    chart_data["share"] = chart_data["stories"].where(daily_totals > 0, 0) / daily_totals.where(daily_totals > 0, 1)
    return (
        alt.Chart(chart_data)
        .mark_area(opacity=0.78, interpolate="monotone")
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %d", labelAngle=-35, tickCount=6, grid=False)),
            y=alt.Y(
                "share:Q",
                title="Share of stories",
                stack="normalize",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1], grid=True),
            ),
            color=alt.Color(
                "legend_issue:N",
                title=None,
                scale=alt.Scale(
                    domain=[ISSUE_LEGEND_LABELS[issue] for issue in ISSUE_AREA_COLORS],
                    range=list(ISSUE_AREA_COLORS.values()),
                ),
                legend=alt.Legend(
                    orient="bottom",
                    direction="horizontal",
                    columns=5,
                    labelLimit=140,
                    symbolSize=92,
                    symbolStrokeWidth=0,
                    labelFontSize=11,
                ),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("classified_issue_area:N", title="Issue"),
                alt.Tooltip("stories:Q", title="Stories"),
                alt.Tooltip("share:Q", title="Share", format=".0%"),
            ],
        )
        .properties(height=315)
        .configure_axisY(titlePadding=14, labelPadding=6)
    )


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
        source_name = escape(str(row.get("source_display", source_display_name(row))))
        url = str(row.get("url", "")).strip()
        if url and url.lower() not in {"nan", "none", ""}:
            source = f'<a class="story-source-link" href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{source_name}</a>'
        else:
            source = source_name
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
    st.session_state.setdefault("overview_regions", [])
    selected_geos = st.multiselect("NY regions", options=NY_REGIONS, key="overview_regions")

    base = filtered_for_overview(df, selected_geos, [])
    if base.empty:
        st.warning("No stories match the selected filters.")
        return

    st.markdown("**Issue trends over time**")
    prior_issue = st.session_state.get("selected_overview_issue")
    if prior_issue and prior_issue not in set(base["classified_issue_area"]):
        prior_issue = None
        st.session_state["selected_overview_issue"] = None
    chart_dates = pd.to_datetime(base["date"], errors="coerce").dropna()
    chart_start = chart_dates.min().normalize()
    chart_end = chart_dates.max().normalize()
    tone_reference = base.groupby("classified_issue_area")["sentiment_score"].mean()

    st.write("This shows what share of daily discussion each issue is taking up.")
    st.altair_chart(topic_share_chart(base, chart_start, chart_end), width="stretch")
    st.markdown("### See Social Trends By Topic")
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

    issue_order = sorted(df["classified_issue_area"].dropna().unique())
    visible_issues = [prior_issue] if prior_issue else issue_order
    for issue in visible_issues:
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
            issue_mini_chart(base, issue, prior_issue, chart_start, chart_end, tone_reference),
            width="stretch",
            key=chart_key,
            on_select="rerun",
            selection_mode=[f"issue_select_{issue_key(issue)}"],
        )
        chart_issue = selected_issue_from_mini_state(chart_state)
        if chart_issue:
            st.session_state["selected_overview_issue"] = chart_issue
            st.rerun()

        if prior_issue:
            st.caption(f"Showing stories for {issue_label(prior_issue)}. Line color reflects whether coverage is concerned, mixed, or constructive.")
            st.caption(
                "Coverage tone is a simple reading of public language: concerned, mixed, or constructive."
            )
            st.write("**Stories driving the movement**")
            render_story_table(filtered_for_overview(base, [], [], selected_issue=prior_issue))
            return

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
    weekly_table = outputs["weekly_issue_brief_table"].dropna(how="all").reset_index(drop=True)
    weekly_height = max(120, min(320, 44 + 38 * len(weekly_table)))

    st.write("**Weekly issue brief**")
    st.dataframe(
        weekly_table,
        hide_index=True,
        width="stretch",
        height=weekly_height,
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
        "Social listening turns the news cycle into structured context for experimentation. As new stories take "
        "up more attention, campaign teams can test message frames that match the current issue environment, "
        "then adapt future outreach based on what is actually resonating."
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
        "Rewards are research and engagement signals. This prototype does not claim measured persuasion effects."
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
        grouped into six New York regions, and converted into human research outputs. Live public sources can be
        sparse for narrowly filtered New York political narratives, so this demo uses GDELT and Reddit as grounding
        signals and extends them into a realistic 14-day monitoring feed for demonstration.

        **Data sources:** GDELT public news and Reddit public posts.
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
    days_back = 14
    if GDELT_PATH.exists() or REDDIT_PATH.exists():
        try:
            df = load_monitoring_feed(
                str(GDELT_PATH) if GDELT_PATH.exists() else None,
                str(REDDIT_PATH) if REDDIT_PATH.exists() else None,
                days_back,
            )
            if df.empty:
                st.warning("No public-source rows are available in the selected time period.")
        except Exception as exc:
            st.warning(f"Could not load public-source caches. {exc}")
            df = pd.DataFrame()
    else:
        st.warning("Public-source data files were not found.")
        df = pd.DataFrame()

    if df.empty:
        st.title("Social Listening")
        st.info("No public-source rows are available for the selected date range.")
        return

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
        render_floating_clear_filters()
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
