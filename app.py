"""Tender Intelligence Decision Support - Streamlit MVP."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "polifarma_synthetic_tenders_2021_2025.csv"

HISTORICAL_TEXT_FIELDS = [
    "tender_title",
    "product_name",
    "product_group",
    "buyer_institution",
    "region",
    "procedure_type",
]
QUERY_TEXT_FIELDS = ["product_name", "product_group", "region", "procedure_type"]

SIMILARITY_WEIGHTS = {
    "tfidf": 0.60,
    "product_group": 0.15,
    "product_name": 0.10,
    "region": 0.10,
    "procedure_type": 0.05,
}


st.set_page_config(
    page_title="Tender Intelligence Decision Support",
    page_icon="TI",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --bg: #0b1020;
            --panel: #111a2d;
            --panel-2: #0f1728;
            --border: rgba(148, 163, 184, 0.20);
            --text: #eef4fb;
            --muted: #95a5bb;
            --blue: #3b82f6;
            --cyan: #22c7c9;
            --amber: #f59e0b;
            --green: #22c55e;
            --red: #ef4444;
        }

        .stApp {
            background:
                radial-gradient(circle at 82% -12%, rgba(34, 199, 201, 0.12), transparent 32%),
                linear-gradient(180deg, #0b1020 0%, #090e1a 100%);
            color: var(--text);
        }

        [data-testid="stHeader"] { background: transparent; }
        .block-container {
            max-width: 1500px;
            padding-top: 1.4rem;
            padding-bottom: 2.4rem;
        }

        [data-testid="stSidebar"] {
            background: #080d18;
            border-right: 1px solid var(--border);
        }

        .brand-mark {
            width: 38px;
            height: 38px;
            display: grid;
            place-items: center;
            border-radius: 8px;
            background: linear-gradient(135deg, var(--blue), var(--cyan));
            color: white;
            font-weight: 800;
            margin-bottom: 0.8rem;
        }

        .sidebar-title {
            color: var(--text);
            font-size: 1.05rem;
            font-weight: 800;
            letter-spacing: 0.04em;
        }

        .sidebar-note {
            color: var(--muted);
            font-size: 0.76rem;
            line-height: 1.45;
            margin-top: 0.45rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }

        .eyebrow {
            color: #7dd3fc;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .page-title {
            color: var(--text);
            font-size: clamp(2rem, 3vw, 3rem);
            line-height: 1.02;
            font-weight: 790;
            margin: 0;
        }

        .page-subtitle {
            color: var(--muted);
            max-width: 840px;
            font-size: 0.98rem;
            margin-top: 0.7rem;
            line-height: 1.55;
        }

        .scope-pill {
            display: inline-flex;
            gap: 0.42rem;
            align-items: center;
            float: right;
            margin-top: 0.5rem;
            padding: 0.42rem 0.7rem;
            border: 1px solid rgba(34, 197, 94, 0.26);
            border-radius: 999px;
            color: #86efac;
            background: rgba(34, 197, 94, 0.08);
            font-size: 0.72rem;
            font-weight: 800;
        }

        .scope-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--green);
            box-shadow: 0 0 14px rgba(34, 197, 94, 0.7);
        }

        .section-kicker {
            color: #7dd3fc;
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.18rem;
        }

        .section-title {
            color: var(--text);
            font-size: 1.08rem;
            font-weight: 760;
            margin-bottom: 0.9rem;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(145deg, rgba(17, 26, 45, 0.98), rgba(13, 20, 35, 0.98));
            border: 1px solid var(--border);
            border-radius: 8px;
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.18);
        }

        [data-testid="stTextInput"] label,
        [data-testid="stNumberInput"] label,
        [data-testid="stSelectbox"] label {
            color: #b9c7d8 !important;
            font-size: 0.78rem !important;
            font-weight: 650 !important;
        }

        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] > div > div {
            background: #0b1323;
            color: var(--text);
            border-color: var(--border);
            border-radius: 8px;
        }

        [data-testid="stButton"] button {
            width: 100%;
            min-height: 42px;
            border: 0;
            border-radius: 8px;
            background: linear-gradient(135deg, var(--amber), #fb923c);
            color: #1f1300;
            font-weight: 850;
            margin-top: 1.6rem;
        }

        .metric-card {
            min-height: 128px;
            padding: 1rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: rgba(15, 23, 40, 0.72);
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .metric-value {
            color: var(--text);
            font-size: 1.55rem;
            font-weight: 820;
            margin-top: 0.45rem;
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.72rem;
            margin-top: 0.3rem;
        }

        .score-card {
            padding: 1.2rem;
            border: 1px solid rgba(34, 197, 94, 0.28);
            border-radius: 8px;
            background: linear-gradient(145deg, rgba(34, 197, 94, 0.12), rgba(59, 130, 246, 0.08));
            min-height: 300px;
        }

        .score-value {
            color: var(--text);
            font-size: 4.2rem;
            line-height: 0.95;
            font-weight: 850;
            letter-spacing: 0;
        }

        .score-label {
            color: #bbf7d0;
            font-size: 1.1rem;
            font-weight: 800;
            margin-top: 0.6rem;
        }

        .confidence {
            display: inline-flex;
            align-items: center;
            padding: 0.32rem 0.58rem;
            border-radius: 999px;
            background: rgba(59, 130, 246, 0.12);
            border: 1px solid rgba(59, 130, 246, 0.24);
            color: #bfdbfe;
            font-size: 0.72rem;
            font-weight: 800;
        }

        .explain-box {
            padding: 1rem;
            border: 1px solid rgba(125, 211, 252, 0.20);
            border-radius: 8px;
            background: rgba(14, 23, 42, 0.66);
            color: #c9d7e8;
            line-height: 1.55;
            font-size: 0.92rem;
        }

        .scope-note {
            padding: 0.85rem 0.95rem;
            border: 1px solid rgba(245, 158, 11, 0.22);
            border-radius: 8px;
            background: rgba(245, 158, 11, 0.07);
            color: #e9c99d;
            font-size: 0.78rem;
            line-height: 1.45;
        }

        @media (max-width: 900px) {
            .scope-pill { float: none; margin-top: 1rem; }
            .score-value { font-size: 3rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_text(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def tokenize(value: Any) -> set[str]:
    text = normalize_text(value)
    return set(re.findall(r"[\w.%]+", text, flags=re.UNICODE))


def combine_text(record: pd.Series | dict[str, Any], fields: list[str]) -> str:
    return " | ".join(normalize_text(record.get(field, "")) for field in fields)


def exact_match_score(left: Any, right: Any) -> float:
    return 1.0 if normalize_text(left) == normalize_text(right) else 0.0


def product_name_score(query_name: str, candidate_name: str) -> float:
    query_normalized = normalize_text(query_name)
    candidate_normalized = normalize_text(candidate_name)
    if not query_normalized or not candidate_normalized:
        return 0.0
    if query_normalized == candidate_normalized:
        return 1.0
    if query_normalized in candidate_normalized or candidate_normalized in query_normalized:
        return 1.0

    query_tokens = tokenize(query_name)
    candidate_tokens = tokenize(candidate_name)
    if not query_tokens or not candidate_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / max(len(query_tokens), len(candidate_tokens))


def hybrid_similarity_score(
    tfidf_score: float,
    product_group: float,
    product_name: float,
    region: float,
    procedure_type: float,
) -> float:
    return (
        SIMILARITY_WEIGHTS["tfidf"] * tfidf_score
        + SIMILARITY_WEIGHTS["product_group"] * product_group
        + SIMILARITY_WEIGHTS["product_name"] * product_name
        + SIMILARITY_WEIGHTS["region"] * region
        + SIMILARITY_WEIGHTS["procedure_type"] * procedure_type
    )


@st.cache_data
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    required = [
        *HISTORICAL_TEXT_FIELDS,
        "tender_id",
        "year",
        "winning_unit_price_try",
        "inflation_adjusted_unit_price_2025_try",
        "gross_margin_pct",
        "discount_to_estimated_cost_pct",
        "strategic_fit_score",
    ]
    missing = [field for field in required if field not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


@st.cache_resource
def build_tfidf_index(search_texts: tuple[str, ...]) -> tuple[TfidfVectorizer, Any]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(search_texts)
    return vectorizer, matrix


def get_similarity_index(df: pd.DataFrame) -> tuple[TfidfVectorizer, Any]:
    search_texts = tuple(df.apply(lambda row: combine_text(row, HISTORICAL_TEXT_FIELDS), axis=1))
    return build_tfidf_index(search_texts)


def retrieve_similar_tenders(df: pd.DataFrame, query: dict[str, Any]) -> pd.DataFrame:
    vectorizer, matrix = get_similarity_index(df)
    query_text = combine_text(query, QUERY_TEXT_FIELDS)
    query_vector = vectorizer.transform([query_text])
    tfidf_scores = cosine_similarity(query_vector, matrix)[0]
    top_indices = tfidf_scores.argsort()[::-1][:30]

    candidates = []
    for initial_rank, idx in enumerate(top_indices, start=1):
        row = df.iloc[int(idx)]
        tfidf_score = float(tfidf_scores[idx])
        group_score = exact_match_score(query["product_group"], row["product_group"])
        name_score = product_name_score(query["product_name"], row["product_name"])
        region_score = exact_match_score(query["region"], row["region"])
        procedure_score = exact_match_score(query["procedure_type"], row["procedure_type"])
        final_score = hybrid_similarity_score(
            tfidf_score,
            group_score,
            name_score,
            region_score,
            procedure_score,
        )

        item = row.to_dict()
        item.update(
            {
                "initial_tfidf_rank": initial_rank,
                "tfidf_score": tfidf_score,
                "product_group_score": group_score,
                "product_name_score": name_score,
                "region_score": region_score,
                "procedure_type_score": procedure_score,
                "final_similarity_score": final_score,
            }
        )
        candidates.append(item)

    return (
        pd.DataFrame(candidates)
        .sort_values(
            [
                "final_similarity_score",
                "tfidf_score",
                "product_name_score",
                "region_score",
                "procedure_type_score",
            ],
            ascending=False,
        )
        .head(10)
        .reset_index(drop=True)
    )


def percentile_metrics(series: pd.Series) -> dict[str, float]:
    return {
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "max": float(series.max()),
        "average": float(series.mean()),
        "std": float(series.std(ddof=0)),
    }


def confidence_level(count: int, average_similarity: float) -> str:
    if count < 7 or average_similarity < 0.55:
        return "Low"
    if count == 10 and average_similarity >= 0.75:
        return "High"
    return "Medium"


def scenario_margins(corridor: dict[str, float], cost: float) -> dict[str, float]:
    scenarios = {
        "Conservative": corridor["p25"],
        "Balanced": corridor["median"],
        "Aggressive": corridor["p75"],
    }
    return {
        name: ((price - cost) / price * 100) if price > 0 else 0
        for name, price in scenarios.items()
    }


def margin_score(margin_pct: float) -> int:
    if margin_pct >= 25:
        return 100
    if margin_pct >= 15:
        return 75
    if margin_pct >= 8:
        return 50
    if margin_pct >= 0:
        return 25
    return 0


def competition_score(competitor_count: int) -> int:
    if competitor_count <= 2:
        return 90
    if competitor_count <= 4:
        return 70
    if competitor_count <= 6:
        return 45
    return 25


def delivery_score(delivery_months: int) -> int:
    return {3: 60, 6: 75, 9: 85, 12: 90}.get(delivery_months, 75)


def attractiveness_label(score: float) -> str:
    if score >= 75:
        return "Strong Opportunity"
    if score >= 60:
        return "Proceed with Balanced Pricing"
    if score >= 45:
        return "Proceed with Margin Protection"
    return "Manual Review Required"


def format_try(value: float) -> str:
    return f"{value:,.2f} TL".replace(",", "X").replace(".", ",").replace("X", ".")


def format_pct(value: float) -> str:
    return f"%{value:.1f}"


def build_gauge(score: float) -> go.Figure:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 46, "color": "#eef4fb"}, "suffix": "/100"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                "bar": {"color": "#22c55e", "thickness": 0.26},
                "bgcolor": "rgba(255,255,255,0.04)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 45], "color": "rgba(239,68,68,0.18)"},
                    {"range": [45, 60], "color": "rgba(245,158,11,0.18)"},
                    {"range": [60, 75], "color": "rgba(59,130,246,0.18)"},
                    {"range": [75, 100], "color": "rgba(34,197,94,0.18)"},
                ],
            },
        )
    )
    figure.update_layout(
        height=230,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#eef4fb"},
    )
    return figure


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


df = load_dataset()

with st.sidebar:
    st.markdown(
        """
        <div class="brand-mark">TI</div>
        <div class="sidebar-title">TENDER IQ</div>
        <div class="sidebar-note">
            Tender Intelligence Decision Support for historical won-tender
            benchmarking. No outcome prediction is used.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Dataset")
    st.write(f"{len(df):,} historical won tenders".replace(",", "."))
    st.write(f"{df['year'].min()}-{df['year'].max()}")
    st.write("Polifarma synthetic tender memory")

header_left, header_right = st.columns([5, 1.4])
with header_left:
    st.markdown(
        """
        <div class="eyebrow">Tender Intelligence Decision Support</div>
        <h1 class="page-title">Historical Tender Benchmarking</h1>
        <p class="page-subtitle">
            Find similar historical won tenders, benchmark achieved prices in
            2025 TRY terms, simulate margin scenarios, and score tender
            attractiveness from explainable business signals.
        </p>
        """,
        unsafe_allow_html=True,
    )
with header_right:
    st.markdown(
        """
        <div class="scope-pill"><span class="scope-dot"></span> WON TENDER MEMORY</div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="scope-note">
        This MVP is a historical benchmarking and price recommendation engine.
        The dataset contains only historical won tenders, so the app does not
        estimate award likelihood or classify outcomes.
    </div>
    """,
    unsafe_allow_html=True,
)

product_names = sorted(df["product_name"].dropna().unique())
product_groups = sorted(df["product_group"].dropna().unique())
regions = sorted(df["region"].dropna().unique())
procedure_types = sorted(df["procedure_type"].dropna().unique())

with st.container(border=True):
    st.markdown(
        """
        <div class="section-kicker">New Tender</div>
        <div class="section-title">Analysis Inputs</div>
        """,
        unsafe_allow_html=True,
    )
    row_1 = st.columns([1.6, 1.0, 1.0, 1.1], gap="small")
    with row_1[0]:
        product_name = st.selectbox(
            "Product Name",
            product_names,
            index=product_names.index("%0.9 NaCl 500 ml")
            if "%0.9 NaCl 500 ml" in product_names
            else 0,
        )
    with row_1[1]:
        product_group = st.selectbox(
            "Product Group",
            product_groups,
            index=product_groups.index("IV Solution") if "IV Solution" in product_groups else 0,
        )
    with row_1[2]:
        region = st.selectbox(
            "Region",
            regions,
            index=regions.index("Marmara") if "Marmara" in regions else 0,
        )
    with row_1[3]:
        procedure_type = st.selectbox(
            "Procedure Type",
            procedure_types,
            index=procedure_types.index("Açık İhale") if "Açık İhale" in procedure_types else 0,
        )

    row_2 = st.columns([1.0, 1.0, 1.0, 1.0, 0.9], gap="small")
    with row_2[0]:
        estimated_unit_cost_try = st.number_input(
            "Estimated Unit Cost TRY",
            min_value=0.01,
            max_value=10_000.0,
            value=18.00,
            step=0.50,
            format="%.2f",
        )
    with row_2[1]:
        quantity = st.number_input(
            "Quantity",
            min_value=1,
            max_value=5_000_000,
            value=100_000,
            step=1_000,
            format="%d",
        )
    with row_2[2]:
        delivery_months = st.selectbox("Delivery Months", [3, 6, 9, 12], index=1)
    with row_2[3]:
        competitor_count = st.number_input(
            "Estimated Competitor Count",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
        )
    with row_2[4]:
        analyze_clicked = st.button("Analiz Et", type="primary")

if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = False

if analyze_clicked:
    st.session_state.analysis_ready = True

if not st.session_state.analysis_ready:
    st.info("Enter a tender scenario and click Analiz Et to generate historical benchmarks.")
    st.stop()

query = {
    "product_name": product_name,
    "product_group": product_group,
    "region": region,
    "procedure_type": procedure_type,
}

similar = retrieve_similar_tenders(df, query)
price_corridor = percentile_metrics(similar["inflation_adjusted_unit_price_2025_try"])
nominal_reference = percentile_metrics(similar["winning_unit_price_try"])
margin_benchmark = {
    "average": float(similar["gross_margin_pct"].mean()),
    "median": float(similar["gross_margin_pct"].median()),
    "p25": float(similar["gross_margin_pct"].quantile(0.25)),
    "p75": float(similar["gross_margin_pct"].quantile(0.75)),
}
discount_benchmark = {
    "average": float(similar["discount_to_estimated_cost_pct"].mean()),
    "median": float(similar["discount_to_estimated_cost_pct"].median()),
}

scenario_prices = {
    "Conservative": price_corridor["p25"],
    "Balanced": price_corridor["median"],
    "Aggressive": price_corridor["p75"],
}
margins = scenario_margins(price_corridor, estimated_unit_cost_try)
average_similarity = float(similar["final_similarity_score"].mean())
confidence = confidence_level(len(similar), average_similarity)

similarity_component = average_similarity * 100
balanced_margin = margins["Balanced"]
margin_component = margin_score(balanced_margin)
strategic_fit_component = float(similar["strategic_fit_score"].mean())
competition_component = competition_score(int(competitor_count))
delivery_component = delivery_score(int(delivery_months))

final_attractiveness_score = (
    0.25 * similarity_component
    + 0.30 * margin_component
    + 0.20 * strategic_fit_component
    + 0.15 * competition_component
    + 0.10 * delivery_component
)
final_label = attractiveness_label(final_attractiveness_score)

st.markdown("---")

top_metrics = st.columns(4, gap="medium")
with top_metrics[0]:
    metric_card("Similar Won Tenders", str(len(similar)), f"{confidence} confidence")
with top_metrics[1]:
    metric_card("Balanced Price", format_try(scenario_prices["Balanced"]), "Historical median, 2025 TRY")
with top_metrics[2]:
    metric_card("Balanced Margin", format_pct(balanced_margin), "Based on entered unit cost")
with top_metrics[3]:
    metric_card("Avg Similarity", f"{average_similarity:.2f}", "Hybrid retrieval score")

section_1, section_2 = st.columns([1.55, 1.0], gap="medium")

with section_1:
    with st.container(border=True):
        st.markdown(
            """
            <div class="section-kicker">Section 1</div>
            <div class="section-title">Top 10 Similar Historical Won Tenders</div>
            """,
            unsafe_allow_html=True,
        )
        table = similar[
            [
                "tender_id",
                "year",
                "product_name",
                "product_group",
                "region",
                "procedure_type",
                "winning_unit_price_try",
                "inflation_adjusted_unit_price_2025_try",
                "gross_margin_pct",
                "discount_to_estimated_cost_pct",
                "final_similarity_score",
            ]
        ].copy()
        table.columns = [
            "Tender ID",
            "Year",
            "Product",
            "Group",
            "Region",
            "Procedure",
            "Winning Unit Price",
            "Adj. Unit Price 2025",
            "Gross Margin %",
            "Discount %",
            "Similarity",
        ]
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "Winning Unit Price": st.column_config.NumberColumn(format="%.2f TL"),
                "Adj. Unit Price 2025": st.column_config.NumberColumn(format="%.2f TL"),
                "Gross Margin %": st.column_config.NumberColumn(format="%.2f"),
                "Discount %": st.column_config.NumberColumn(format="%.2f"),
                "Similarity": st.column_config.NumberColumn(format="%.3f"),
            },
        )

with section_2:
    with st.container(border=True):
        st.markdown(
            """
            <div class="section-kicker">Section 4</div>
            <div class="section-title">Tender Attractiveness Score</div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_gauge(final_attractiveness_score),
            config={"displayModeBar": False, "staticPlot": True},
        )
        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-value">{final_attractiveness_score:.0f}</div>
                <div class="score-label">{final_label}</div>
                <p class="metric-note">
                    Components: similarity {similarity_component:.0f},
                    margin {margin_component}, strategic fit {strategic_fit_component:.0f},
                    competition {competition_component}, delivery {delivery_component}.
                </p>
                <span class="confidence">{confidence} Confidence</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

section_3, section_4, section_5 = st.columns(3, gap="medium")

with section_3:
    with st.container(border=True):
        st.markdown(
            """
            <div class="section-kicker">Section 2</div>
            <div class="section-title">Historical Price Corridor</div>
            """,
            unsafe_allow_html=True,
        )
        corridor_rows = pd.DataFrame(
            [
                ["Min", price_corridor["min"], nominal_reference["min"]],
                ["P25", price_corridor["p25"], nominal_reference["p25"]],
                ["Median", price_corridor["median"], nominal_reference["median"]],
                ["P75", price_corridor["p75"], nominal_reference["p75"]],
                ["Max", price_corridor["max"], nominal_reference["max"]],
                ["Average", price_corridor["average"], nominal_reference["average"]],
                ["Std. Dev.", price_corridor["std"], nominal_reference["std"]],
            ],
            columns=["Metric", "2025 TRY Benchmark", "Nominal Reference"],
        )
        st.dataframe(
            corridor_rows,
            hide_index=True,
            width="stretch",
            column_config={
                "2025 TRY Benchmark": st.column_config.NumberColumn(format="%.2f TL"),
                "Nominal Reference": st.column_config.NumberColumn(format="%.2f TL"),
            },
        )
        st.markdown("##### Price Recommendation")
        st.dataframe(
            pd.DataFrame(
                [
                    ["Conservative", scenario_prices["Conservative"]],
                    ["Balanced", scenario_prices["Balanced"]],
                    ["Aggressive", scenario_prices["Aggressive"]],
                ],
                columns=["Scenario", "Recommended Unit Price"],
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "Recommended Unit Price": st.column_config.NumberColumn(format="%.2f TL")
            },
        )

with section_4:
    with st.container(border=True):
        st.markdown(
            """
            <div class="section-kicker">Section 3</div>
            <div class="section-title">Margin Simulation</div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(
            pd.DataFrame(
                [
                    ["Conservative", scenario_prices["Conservative"], margins["Conservative"]],
                    ["Balanced", scenario_prices["Balanced"], margins["Balanced"]],
                    ["Aggressive", scenario_prices["Aggressive"], margins["Aggressive"]],
                ],
                columns=["Scenario", "Unit Price", "Margin %"],
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "Unit Price": st.column_config.NumberColumn(format="%.2f TL"),
                "Margin %": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.markdown("##### Historical Benchmark Margin")
        st.dataframe(
            pd.DataFrame(
                [
                    ["Average gross_margin_pct", margin_benchmark["average"]],
                    ["Median gross_margin_pct", margin_benchmark["median"]],
                    ["P25 gross_margin_pct", margin_benchmark["p25"]],
                    ["P75 gross_margin_pct", margin_benchmark["p75"]],
                ],
                columns=["Metric", "Value"],
            ),
            hide_index=True,
            width="stretch",
            column_config={"Value": st.column_config.NumberColumn(format="%.2f")},
        )

with section_5:
    with st.container(border=True):
        st.markdown(
            """
            <div class="section-kicker">Benchmark Detail</div>
            <div class="section-title">Historical Discount Benchmark</div>
            """,
            unsafe_allow_html=True,
        )
        metric_card(
            "Average Discount",
            format_pct(discount_benchmark["average"]),
            "Top 10 similar won tenders",
        )
        st.write("")
        metric_card(
            "Median Discount",
            format_pct(discount_benchmark["median"]),
            "Discount to estimated cost",
        )
        st.write("")
        metric_card(
            "Estimated Contract Value",
            format_try(scenario_prices["Balanced"] * quantity),
            "Balanced price x quantity",
        )

with st.container(border=True):
    st.markdown(
        """
        <div class="section-kicker">Section 5</div>
        <div class="section-title">Business Explanation</div>
        """,
        unsafe_allow_html=True,
    )
    explanation = (
        f"{len(similar)} similar historical won tenders were found with {confidence.lower()} confidence. "
        f"The inflation-adjusted historical median price is {format_try(scenario_prices['Balanced'])}. "
        f"Based on the estimated unit cost of {format_try(estimated_unit_cost_try)}, the balanced scenario "
        f"produces a margin of {format_pct(balanced_margin)}. The tender receives a Tender Attractiveness "
        f"Score of {final_attractiveness_score:.0f}/100 and is classified as {final_label}."
    )
    st.markdown(f'<div class="explain-box">{explanation}</div>', unsafe_allow_html=True)
