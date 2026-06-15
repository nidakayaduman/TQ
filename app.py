"""Tender Intelligence Platform - Streamlit presentation demo."""

import time
from statistics import mean

import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Tender Intelligence Platform",
    page_icon="TI",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Global styles
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        :root {
            --navy: #0A0F1E;
            --navy-soft: #10182A;
            --navy-card: #121C30;
            --navy-card-2: #16223A;
            --blue: #2D7DD2;
            --blue-soft: #69A7E8;
            --amber: #F4A261;
            --green: #39C58A;
            --text: #F3F7FC;
            --muted: #8FA2BD;
            --border: rgba(117, 151, 193, 0.18);
        }

        .stApp {
            background:
                radial-gradient(circle at 75% -10%, rgba(45, 125, 210, 0.13), transparent 30%),
                var(--navy);
            color: var(--text);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0D1424 0%, #090E1A 100%);
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--muted);
        }

        [data-testid="stSidebarCollapseButton"] {
            color: var(--muted);
        }

        .block-container {
            max-width: 1600px;
            padding-top: 1.6rem;
            padding-bottom: 2.5rem;
        }

        h1, h2, h3, p {
            font-family: "Inter", "Segoe UI", sans-serif;
        }

        .eyebrow {
            color: var(--blue-soft);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .page-title {
            color: var(--text);
            font-size: clamp(1.8rem, 3vw, 2.75rem);
            font-weight: 720;
            letter-spacing: -0.045em;
            line-height: 1.04;
            margin: 0;
        }

        .page-subtitle {
            color: var(--muted);
            font-size: 0.95rem;
            margin-top: 0.6rem;
            max-width: 720px;
        }

        .live-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.42rem 0.72rem;
            border: 1px solid rgba(57, 197, 138, 0.25);
            border-radius: 999px;
            background: rgba(57, 197, 138, 0.08);
            color: #8AE6BD;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            float: right;
            margin-top: 0.5rem;
        }

        .live-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--green);
            box-shadow: 0 0 10px rgba(57, 197, 138, 0.8);
        }

        .section-kicker {
            color: var(--blue-soft);
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.2rem;
        }

        .section-title {
            color: var(--text);
            font-size: 1.08rem;
            font-weight: 680;
            margin: 0 0 0.85rem 0;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(135deg, rgba(18, 28, 48, 0.98), rgba(14, 23, 40, 0.98));
            border: 1px solid var(--border);
            border-radius: 14px;
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.18);
        }

        [data-testid="stVerticalBlockBorderWrapper"] > div {
            padding: 0.15rem 0.2rem;
        }

        .panel-shell {
            background: linear-gradient(145deg, rgba(18, 28, 48, 0.96), rgba(13, 21, 37, 0.96));
            border: 1px solid var(--border);
            border-radius: 14px;
            min-height: 445px;
            padding: 1rem 1rem 0.8rem;
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.16);
        }

        [data-testid="stTextInput"] label,
        [data-testid="stNumberInput"] label,
        [data-testid="stSelectbox"] label,
        [data-testid="stSlider"] label {
            color: #AEC0D8 !important;
            font-size: 0.77rem !important;
            font-weight: 600 !important;
        }

        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] > div > div {
            background: #0C1424;
            color: var(--text);
            border-color: var(--border);
            border-radius: 8px;
        }

        [data-testid="stSlider"] [role="slider"] {
            background-color: var(--amber);
            border-color: var(--amber);
        }

        [data-testid="stButton"] {
            padding-top: 1.55rem;
        }

        [data-testid="stButton"] button {
            width: 100%;
            min-height: 42px;
            border: 0;
            border-radius: 8px;
            background: linear-gradient(135deg, #F4A261, #EE8D46);
            color: #17100A;
            font-weight: 800;
            box-shadow: 0 8px 22px rgba(244, 162, 97, 0.2);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }

        [data-testid="stButton"] button:hover {
            transform: translateY(-1px);
            color: #17100A;
            box-shadow: 0 11px 28px rgba(244, 162, 97, 0.3);
        }

        .tender-table-wrap {
            overflow-x: auto;
            border: 1px solid rgba(117, 151, 193, 0.12);
            border-radius: 10px;
        }

        .tender-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.69rem;
            white-space: nowrap;
        }

        .tender-table th {
            background: #0D1627;
            color: #8296B1;
            font-size: 0.61rem;
            font-weight: 700;
            letter-spacing: 0.045em;
            padding: 0.72rem 0.48rem;
            text-align: left;
            text-transform: uppercase;
        }

        .tender-table td {
            color: #DCE6F3;
            padding: 0.78rem 0.48rem;
            border-top: 1px solid rgba(117, 151, 193, 0.1);
        }

        .tender-table tr:hover td {
            background: rgba(45, 125, 210, 0.045);
        }

        .win-badge, .risk-badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            font-weight: 700;
        }

        .win-badge {
            background: rgba(57, 197, 138, 0.11);
            border: 1px solid rgba(57, 197, 138, 0.24);
            color: #77DFB0;
            font-size: 0.61rem;
            padding: 0.23rem 0.42rem;
        }

        .corridor-values {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.35rem;
            margin: 1.35rem 0 0.55rem;
        }

        .corridor-value {
            color: var(--muted);
            font-size: 0.61rem;
        }

        .corridor-value strong {
            display: block;
            color: var(--text);
            font-size: 0.9rem;
            margin-top: 0.18rem;
        }

        .corridor-value:nth-child(2) { text-align: center; }
        .corridor-value:nth-child(2) strong { color: var(--amber); }
        .corridor-value:nth-child(3) { text-align: right; }

        .price-track-wrap {
            position: relative;
            height: 82px;
            margin: 0.1rem 0 0.35rem;
        }

        .price-track {
            position: absolute;
            top: 32px;
            left: 2%;
            width: 96%;
            height: 10px;
            border-radius: 999px;
            background: linear-gradient(90deg, #2D7DD2 0%, #48A9A6 42%, #F4A261 68%, #E76F51 100%);
            box-shadow: 0 0 20px rgba(45, 125, 210, 0.17);
        }

        .track-marker {
            position: absolute;
            top: 22px;
            width: 2px;
            height: 30px;
            background: #EAF2FC;
        }

        .track-marker::after {
            content: "";
            position: absolute;
            top: -3px;
            left: -4px;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #EAF2FC;
            box-shadow: 0 0 0 4px rgba(234, 242, 252, 0.12);
        }

        .marker-low { left: 8%; }
        .marker-mid { left: 53%; background: var(--amber); }
        .marker-mid::after { background: var(--amber); box-shadow: 0 0 0 5px rgba(244, 162, 97, 0.16); }
        .marker-high { left: 91%; }

        .insight-card {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.72rem 0;
            border-top: 1px solid rgba(117, 151, 193, 0.11);
        }

        .insight-label {
            color: var(--muted);
            font-size: 0.71rem;
        }

        .insight-value {
            color: var(--text);
            font-size: 0.92rem;
            font-weight: 750;
        }

        .insight-value.amber {
            color: var(--amber);
        }

        .risk-badge {
            background: rgba(244, 162, 97, 0.1);
            border: 1px solid rgba(244, 162, 97, 0.26);
            color: #FFC28E;
            font-size: 0.68rem;
            padding: 0.32rem 0.58rem;
        }

        .mini-card-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.5rem;
            margin-top: -0.2rem;
        }

        .mini-card {
            background: rgba(11, 20, 35, 0.74);
            border: 1px solid rgba(117, 151, 193, 0.12);
            border-radius: 9px;
            padding: 0.68rem 0.55rem;
            min-height: 73px;
        }

        .mini-label {
            color: var(--muted);
            font-size: 0.6rem;
            line-height: 1.25;
        }

        .mini-value {
            color: var(--text);
            font-size: 0.86rem;
            font-weight: 760;
            margin-top: 0.42rem;
        }

        .performance-wrap {
            margin-top: 1.6rem;
            padding-top: 1.15rem;
            border-top: 1px solid rgba(117, 151, 193, 0.15);
        }

        .metric-card {
            position: relative;
            overflow: hidden;
            min-height: 154px;
            background: linear-gradient(145deg, #121C30 0%, #0E1728 100%);
            border: 1px solid var(--border);
            border-radius: 13px;
            padding: 1.05rem 1.1rem;
        }

        .metric-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, var(--amber), transparent 68%);
        }

        .metric-number {
            color: var(--amber);
            font-size: 2rem;
            font-weight: 780;
            letter-spacing: -0.04em;
            line-height: 1;
        }

        .metric-label {
            color: var(--text);
            font-size: 0.79rem;
            font-weight: 680;
            margin-top: 0.68rem;
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.65rem;
            margin-top: 0.3rem;
        }

        .metric-trend {
            position: absolute;
            top: 1.05rem;
            right: 1rem;
            color: var(--green);
            font-size: 0.68rem;
            font-weight: 700;
        }

        .sidebar-brand {
            padding: 0.7rem 0.15rem 1.4rem;
            border-bottom: 1px solid var(--border);
        }

        .sidebar-mark {
            display: inline-grid;
            place-items: center;
            width: 36px;
            height: 36px;
            margin-bottom: 0.8rem;
            background: linear-gradient(135deg, var(--blue), #174E91);
            border-radius: 9px;
            box-shadow: 0 8px 24px rgba(45, 125, 210, 0.25);
            color: white;
            font-size: 0.76rem;
            font-weight: 800;
        }

        .sidebar-title {
            color: var(--text);
            font-size: 1.18rem;
            font-weight: 800;
            letter-spacing: 0.04em;
        }

        .sidebar-subtitle {
            color: var(--muted);
            font-size: 0.72rem;
            line-height: 1.45;
            margin-top: 0.35rem;
        }

        .sidebar-nav {
            margin-top: 1.2rem;
        }

        .nav-item {
            color: var(--muted);
            font-size: 0.77rem;
            padding: 0.72rem 0.78rem;
            border-radius: 8px;
            margin-bottom: 0.25rem;
        }

        .nav-item.active {
            color: #DCEBFB;
            background: rgba(45, 125, 210, 0.12);
            border-left: 2px solid var(--blue);
        }

        .sidebar-footer {
            position: fixed;
            bottom: 1.5rem;
            width: 12rem;
        }

        .partner-tag {
            display: inline-block;
            border: 1px solid rgba(244, 162, 97, 0.24);
            border-radius: 7px;
            background: rgba(244, 162, 97, 0.06);
            color: #DDB68F;
            font-size: 0.64rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            padding: 0.5rem 0.68rem;
            text-transform: uppercase;
        }

        .empty-state {
            margin-top: 1.6rem;
            border: 1px dashed rgba(117, 151, 193, 0.22);
            border-radius: 14px;
            padding: 2.2rem;
            text-align: center;
            color: var(--muted);
            background: rgba(18, 28, 48, 0.35);
        }

        .empty-state strong {
            display: block;
            color: #C7D5E7;
            font-size: 0.9rem;
            margin-bottom: 0.35rem;
        }

        .mvp-disclaimer {
            margin-top: 1rem;
            padding: 0.8rem 0.95rem;
            border: 1px solid rgba(105, 167, 232, 0.2);
            border-radius: 10px;
            background: rgba(45, 125, 210, 0.06);
            color: #9EB2CC;
            font-size: 0.68rem;
            line-height: 1.5;
        }

        .mvp-disclaimer strong {
            color: #C9D8EA;
        }

        [data-testid="stSpinner"] {
            color: var(--amber);
        }

        @media (max-width: 900px) {
            .live-pill { float: none; margin-top: 1rem; }
            .mini-card-grid { grid-template-columns: 1fr; }
            .sidebar-footer { position: static; margin-top: 3rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-mark">TI</div>
            <div class="sidebar-title">TENDER IQ</div>
            <div class="sidebar-subtitle">AI Destekli İhale Karar Platformu</div>
        </div>
        <div class="sidebar-nav">
            <div class="nav-item active">Karar Merkezi</div>
            <div class="nav-item">Geçmiş İhaleler</div>
            <div class="nav-item">Model Performansı</div>
            <div class="nav-item">Veri Kataloğu</div>
        </div>
        <div class="sidebar-footer">
            <div class="partner-tag">Polifarma x EY</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
header_left, header_right = st.columns([5, 1])
with header_left:
    st.markdown(
        """
        <div class="eyebrow">Commercial Excellence / Tender Analytics</div>
        <h1 class="page-title">Tender Intelligence Platform</h1>
        <p class="page-subtitle">
            Geçmiş ihale verilerini ticari önceliklerle birleştiren AI destekli
            fiyatlandırma ve ihale değerlendirme karar merkezi.
        </p>
        """,
        unsafe_allow_html=True,
    )
with header_right:
    st.markdown(
        """
        <div class="live-pill">
            <span class="live-dot"></span>
            MODEL AKTİF
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section 1: New tender input panel
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown(
        """
        <div class="section-kicker">Yeni Analiz</div>
        <div class="section-title">Yeni İhale Girdisi</div>
        """,
        unsafe_allow_html=True,
    )

    input_columns = st.columns([1.45, 1.05, 1.0, 1.0, 1.2, 0.85], gap="small")
    with input_columns[0]:
        product_name = st.text_input("Ürün Adı", value="%0.9 NaCl 500ml")
    with input_columns[1]:
        lot_size = st.number_input(
            "Lot Miktarı",
            min_value=1_000,
            max_value=10_000_000,
            value=1_200_000,
            step=50_000,
            format="%d",
        )
    with input_columns[2]:
        region = st.selectbox(
            "Bölge",
            ["Marmara", "İç Anadolu", "Ege", "Akdeniz", "Karadeniz"],
        )
    with input_columns[3]:
        delivery = st.selectbox(
            "Teslimat Süresi", ["3 ay", "6 ay", "12 ay"], index=1
        )
    with input_columns[4]:
        competitor_count = st.slider("Rakip Sayısı Tahmini", 1, 8, 4)
    with input_columns[5]:
        analyze_clicked = st.button(
            "Analiz Et", type="primary", use_container_width=True
        )

if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = False

if analyze_clicked:
    with st.spinner("İhale senaryosu analiz ediliyor..."):
        time.sleep(0.8)
    st.session_state.analysis_ready = True


# Dummy won-tender history and year-based inflation adjustment assumptions.
INFLATION_FACTORS = {2024: 1.28, 2025: 1.12}
HISTORICAL_TENDERS = [
    {
        "tender_no": "IH-2025-184",
        "year": 2025,
        "product": "%0.9 NaCl 500ml",
        "region": "Marmara",
        "lot": "1.15 mn",
        "winning_unit_price": 10.72,
        "gross_margin": 15.2,
    },
    {
        "tender_no": "IH-2025-141",
        "year": 2025,
        "product": "%0.9 NaCl 500ml",
        "region": "Ege",
        "lot": "980 bin",
        "winning_unit_price": 10.91,
        "gross_margin": 14.6,
    },
    {
        "tender_no": "IH-2024-297",
        "year": 2024,
        "product": "%0.9 NaCl 500ml",
        "region": "İç Anadolu",
        "lot": "1.30 mn",
        "winning_unit_price": 10.58,
        "gross_margin": 13.8,
    },
    {
        "tender_no": "IH-2024-233",
        "year": 2024,
        "product": "%0.9 NaCl 1000ml",
        "region": "Marmara",
        "lot": "760 bin",
        "winning_unit_price": 11.22,
        "gross_margin": 16.1,
    },
    {
        "tender_no": "IH-2024-176",
        "year": 2024,
        "product": "%0.9 NaCl 500ml",
        "region": "Akdeniz",
        "lot": "1.08 mn",
        "winning_unit_price": 10.66,
        "gross_margin": 14.1,
    },
]

for tender in HISTORICAL_TENDERS:
    tender["inflation_factor"] = INFLATION_FACTORS[tender["year"]]
    tender["adjusted_winning_price"] = (
        tender["winning_unit_price"] * tender["inflation_factor"]
    )

adjusted_prices = [
    tender["adjusted_winning_price"] for tender in HISTORICAL_TENDERS
]
recommended_price = mean(adjusted_prices)
lower_price = recommended_price * 0.96
upper_price = recommended_price * 1.04


def build_historical_tender_rows() -> str:
    """Render historical won tenders with nominal and adjusted prices."""
    return "".join(
        (
            f'<tr><td>{tender["tender_no"]}</td>'
            f'<td>{tender["year"]}</td>'
            f'<td>{tender["product"]}</td>'
            f'<td>{tender["region"]}</td>'
            f'<td>{tender["lot"]}</td>'
            f'<td>{tender["winning_unit_price"]:.2f} TL</td>'
            f'<td>{tender["inflation_factor"]:.2f}x</td>'
            f'<td>{tender["adjusted_winning_price"]:.2f} TL</td>'
            f'<td>%{tender["gross_margin"]:.1f}</td>'
            '<td><span class="win-badge">Kazandı</span></td></tr>'
        )
        for tender in HISTORICAL_TENDERS
    )


def build_historical_fit_gauge(value: int) -> go.Figure:
    """Create a gauge representing similarity to historical won tenders."""
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={
                "suffix": "%",
                "font": {"size": 50, "color": "#F3F7FC", "family": "Inter"},
            },
            title={
                "text": "HISTORICAL FIT SCORE",
                "font": {"size": 11, "color": "#8FA2BD", "family": "Inter"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 0,
                    "tickcolor": "rgba(0,0,0,0)",
                    "tickfont": {"color": "rgba(0,0,0,0)", "size": 1},
                },
                "bar": {"color": "#2D7DD2", "thickness": 0.25},
                "bgcolor": "rgba(255,255,255,0.06)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35], "color": "rgba(231,111,81,0.12)"},
                    {"range": [35, 65], "color": "rgba(244,162,97,0.13)"},
                    {"range": [65, 100], "color": "rgba(57,197,138,0.11)"},
                ],
                "threshold": {
                    "line": {"color": "#F4A261", "width": 3},
                    "thickness": 0.8,
                    "value": value,
                },
            },
            domain={"x": [0.08, 0.92], "y": [0.02, 0.98]},
        )
    )
    figure.update_layout(
        height=285,
        margin={"l": 16, "r": 16, "t": 28, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter", "color": "#F3F7FC"},
    )
    return figure


# ---------------------------------------------------------------------------
# Section 2: Analysis result dashboard
# ---------------------------------------------------------------------------
if st.session_state.analysis_ready:
    result_col_1, result_col_2, result_col_3 = st.columns(
        [1.4, 1.0, 1.0], gap="medium"
    )

    with result_col_1:
        with st.container(border=True, height=445):
            st.markdown(
                f"""
                <div class="section-kicker">Referans Veri</div>
                <div class="section-title">Benzer Geçmiş İhaleler</div>
                <div class="tender-table-wrap">
                    <table class="tender-table">
                        <thead>
                            <tr>
                                <th>İhale No</th>
                                <th>Yıl</th>
                                <th>Ürün</th>
                                <th>Bölge</th>
                                <th>Lot</th>
                                <th>Nominal Fiyat</th>
                                <th>Enflasyon K.</th>
                                <th>Düzeltilmiş Fiyat</th>
                                <th>Brüt Marj</th>
                                <th>Sonuç</th>
                            </tr>
                        </thead>
                        <tbody>
                            {build_historical_tender_rows()}
                        </tbody>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with result_col_2:
        with st.container(border=True, height=445):
            st.markdown(
                f"""
                <div class="section-kicker">Fiyat Optimizasyonu</div>
                <div class="section-title">Enflasyona Göre Fiyat Koridoru</div>
                <div class="corridor-values">
                    <div class="corridor-value">Alt Sınır<strong>{lower_price:.2f} TL</strong></div>
                    <div class="corridor-value">Önerilen<strong>{recommended_price:.2f} TL</strong></div>
                    <div class="corridor-value">Üst Sınır<strong>{upper_price:.2f} TL</strong></div>
                </div>
                <div class="price-track-wrap">
                    <div class="price-track"></div>
                    <span class="track-marker marker-low"></span>
                    <span class="track-marker marker-mid"></span>
                    <span class="track-marker marker-high"></span>
                </div>
                <div class="insight-card">
                    <span class="insight-label">Baz Alınan Fiyat</span>
                    <span class="insight-value amber">Düzeltilmiş</span>
                </div>
                <div class="insight-card">
                    <span class="insight-label">Ortalama Enflasyon Katsayısı</span>
                    <span class="insight-value">{mean(INFLATION_FACTORS.values()):.2f}x</span>
                </div>
                <div class="insight-card">
                    <span class="insight-label">Risk Sınıfı</span>
                    <span class="risk-badge">Orta Risk</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with result_col_3:
        with st.container(border=True, height=445):
            st.markdown(
                """
                <div class="section-kicker">Benzerlik Analizi</div>
                <div class="section-title">Geçmiş Kazanım Benzerliği</div>
                """,
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                build_historical_fit_gauge(87),
                use_container_width=True,
                config={"displayModeBar": False, "staticPlot": True},
            )
            st.markdown(
                """
                <div class="mini-card-grid">
                    <div class="mini-card">
                        <div class="mini-label">Referans İhale</div>
                        <div class="mini-value">5</div>
                    </div>
                    <div class="mini-card">
                        <div class="mini-label">Veri Kalitesi</div>
                        <div class="mini-value">Yüksek</div>
                    </div>
                    <div class="mini-card">
                        <div class="mini-label">Senaryo Güveni</div>
                        <div class="mini-value">Orta</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -----------------------------------------------------------------------
    # Section 3: Success metrics dashboard
    # -----------------------------------------------------------------------
    st.markdown(
        """
        <div class="performance-wrap">
            <div class="section-kicker">Backtesting & Pilot Sonuçları</div>
            <div class="section-title">Model Performans Göstergeleri</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4, gap="medium")
    metrics = [
        (
            "%71",
            "Inflation Adjusted Price Corridor Coverage",
            "son 20 ihale backtesting",
            "↑ 8.4 puan",
        ),
        (
            "%82",
            "Similar Tender Expert Approval",
            "uzman değerlendirmesi",
            "↑ 7.2 puan",
        ),
        (
            "%68",
            "Evaluation Time Reduction",
            "manuel vs. sistem",
            "↑ 22 saat",
        ),
        (
            "%64",
            "Recommendation Adoption Rate",
            "pilot kullanım",
            "↑ 6.1 puan",
        ),
    ]

    for column, (value, label, note, trend) in zip(metric_columns, metrics):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-trend">{trend}</div>
                    <div class="metric-number">{value}</div>
                    <div class="metric-label">{label}</div>
                    <div class="metric-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="mvp-disclaimer">
            <strong>MVP kapsam notu:</strong>
            This MVP does not predict win probability. It uses historical won
            tenders to support pricing and tender evaluation. True win
            probability requires both won and lost tender data.
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    st.markdown(
        """
        <div class="empty-state">
            <strong>Karar destek analizi için girdiler hazır.</strong>
            Enflasyona göre fiyat koridoru ve geçmiş kazanılmış ihale benzerliği için
            “Analiz Et” butonunu kullanın.
        </div>
        """,
        unsafe_allow_html=True,
    )
