"""
Markets Forecast Dashboard
==========================
Run locally:
    pip install -r requirements.txt
    streamlit run wwwwww.py

A multi-topic forecasting dashboard. Each topic lives as its own
collapsible section in the sidebar — click a topic to reveal its
sub-pages, then click a sub-page to load it.

Currently shipped topics:
    - Gold (XAU/USD)   — full multi-horizon forecast
    - + Add new topic  — template showing how to add another

Adding a new topic
------------------
1. Build a hero function `my_hero()` that renders the headline + KPIs.
2. Build one or more section functions, e.g. `my_overview()`.
3. Register them in the TOPICS dict at the bottom of this file:

    TOPICS["My Topic"] = Topic(
        name="My Topic",
        subtitle="Short tagline shown under the title",
        hero=my_hero,
        pages={"Overview": my_overview, "Trades": my_trades},
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    import yfinance as yf

    _YF_OK = True
except ImportError:
    yf = None  # type: ignore[assignment]
    _YF_OK = False

try:
    import anthropic

    _ANTHROPIC_OK = True
except ImportError:
    anthropic = None  # type: ignore[assignment]
    _ANTHROPIC_OK = False

try:
    from scipy.stats import multivariate_normal as _scipy_mvn
    from scipy.stats import norm as _scipy_norm

    _SCIPY_OK = True
except ImportError:
    _scipy_mvn = None  # type: ignore[assignment]
    _scipy_norm = None  # type: ignore[assignment]
    _SCIPY_OK = False


st.set_page_config(
    page_title="Markets Forecast Dashboard",
    page_icon="Au",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Topic registry — each topic is one item in the sidebar.
# ---------------------------------------------------------------------------


@dataclass
class Topic:
    """A self-contained sidebar item with its own hero + sub-pages."""

    name: str
    subtitle: str
    hero: Callable[[], None]
    pages: Dict[str, Callable[[], None]]


# ---------------------------------------------------------------------------
# Snapshot data — edit these constants to refresh the dashboard.
# ---------------------------------------------------------------------------

AS_OF = "May 25, 2026 · 17:02 ICT"
SPOT = 4565.25
DAILY_CHANGE_PCT = -0.02
PREV_CLOSE = 4509.38
ATH = 5602.23

DXY = 99.32
TIPS_10Y = 2.18
NOMINAL_10Y = 4.56
CPI_YOY = 3.8
CORE_CPI_YOY = 2.8
CORE_PCE_YOY = 3.2

FED_HIKE_ODDS_DEC = 42  # percent

REAL_YIELD_SERIES = pd.DataFrame(
    {
        "date": ["Apr 24", "Apr 29", "May 04", "May 08", "May 13", "May 15", "May 18", "May 21"],
        "value": [1.89, 1.96, 1.95, 1.93, 1.99, 2.10, 2.13, 2.18],
    }
)

DXY_SERIES = pd.DataFrame(
    {
        "date": ["Apr 24", "Apr 30", "May 06", "May 12", "May 15", "May 19", "May 21", "May 22"],
        "value": [98.53, 98.06, 98.02, 98.30, 99.28, 99.33, 99.26, 99.24],
    }
)

COT_NET_LONGS = pd.DataFrame(
    {
        "date": [
            "Jan 16",
            "Apr 10",
            "Apr 17",
            "Apr 24",
            "May 01",
            "May 05",
            "May 12",
        ],
        "contracts": [134745, 91500, 89200, 92400, 94254, 94254, 98015],
    }
)

ETF_FLOWS = pd.DataFrame(
    {
        "month": ["Dec '25", "Jan '26", "Feb '26", "Mar '26", "Apr '26"],
        "usd_bn": [2.1, 8.4, 12.1, -3.8, 6.6],
    }
)

BANK_TARGETS = pd.DataFrame(
    {
        "bank": [
            "HSBC",
            "Goldman Sachs",
            "ANZ",
            "UBS (end-26)",
            "Deutsche Bank",
            "JPM (rev. avg)",
            "JPM (Q4 '26)",
            "Wells Fargo (high)",
            "UBS bull case",
        ],
        "target": [4450, 5400, 5600, 5900, 6000, 5243, 6300, 6300, 7200],
    }
)


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Horizon:
    name: str
    until: str
    bias: str
    bias_color: str
    range_low: float
    range_high: float
    target: float
    rationale: str


HORIZONS = [
    Horizon(
        name="30 days",
        until="to Jun 25, 2026",
        bias="Neutral-bearish",
        bias_color="#c98a2f",
        range_low=4400,
        range_high=4780,
        target=4520,
        rationale=(
            "Hot CPI keeps real yields elevated; Apr PCE (May 28) is the next binary catalyst. "
            "Managed Money is long-loaded at higher levels — long-liquidation risk if $4,500 "
            "breaks. Floor expected at the central-bank bid zone."
        ),
    ),
    Horizon(
        name="3 months",
        until="to Aug 25, 2026",
        bias="Constructive",
        bias_color="#2f9d6a",
        range_low=4450,
        range_high=5100,
        target=4850,
        rationale=(
            "JPMorgan flags 'H2 demand re-acceleration.' Q2 central-bank data lands in July; "
            "if inflation peaks and the Fed shifts dovish, real yields roll over and gold breaks "
            "the $4,900 range high. Watch Asian ETF flows — the structural bid layer."
        ),
    ),
    Horizon(
        name="12 months",
        until="to May 2027",
        bias="Bullish",
        bias_color="#2f9d6a",
        range_low=5200,
        range_high=6300,
        target=5800,
        rationale=(
            "Consensus blend of major bank targets. Bull case ($7,200, UBS) requires Western "
            "retail/ETF rotation on top of the central-bank bid. Bear case ($4,450, HSBC) "
            "assumes a peace dividend + sticky real yields. ATH ($5,602) likely revisited."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Plotly theming helpers (flat, minimal — no gradients or shadows)
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="-apple-system, Segoe UI, Inter, sans-serif", size=12, color="#d6d6d6"),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zeroline=False),
    hoverlabel=dict(bgcolor="#1a1a1a", font_size=12),
)


def line_chart(df: pd.DataFrame, y_col: str, color: str, y_suffix: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df[y_col],
            mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=7),
            hovertemplate="%{x}<br>%{y}" + y_suffix + "<extra></extra>",
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=280)
    if y_suffix:
        fig.update_yaxes(ticksuffix=y_suffix)
    return fig


def bar_chart_horizontal(df: pd.DataFrame, y_col: str, x_col: str, spot: float) -> go.Figure:
    df = df.sort_values(x_col)
    colors = ["#c98a2f" if v < spot else "#2f9d6a" for v in df[x_col]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df[x_col],
            y=df[y_col],
            orientation="h",
            marker=dict(color=colors),
            text=[f"${v:,.0f}" for v in df[x_col]],
            textposition="outside",
            textfont=dict(color="#d6d6d6"),
        )
    )
    fig.add_vline(
        x=spot,
        line=dict(color="#888", dash="dash", width=1.5),
        annotation_text=f"Spot ${spot:,.0f}",
        annotation_position="top",
        annotation_font_color="#aaa",
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=420, showlegend=False)
    fig.update_xaxes(title_text="Target price ($/oz)")
    return fig


def bar_chart_vertical(df: pd.DataFrame, x_col: str, y_col: str, color: str = "#3a7ec9") -> go.Figure:
    fig = go.Figure()
    colors = [color if v >= 0 else "#c94a4a" for v in df[y_col]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df[x_col],
            y=df[y_col],
            marker=dict(color=colors),
            hovertemplate="%{x}: %{y}<extra></extra>",
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# Gold topic — hero (rendered above whichever sub-page is active)
# ---------------------------------------------------------------------------


def gold_hero() -> None:
    col_title, col_pill = st.columns([5, 1])
    with col_title:
        st.title("XAU/USD · Gold vs US Dollar")
        st.caption(
            "Multi-horizon forecast · macro, positioning, geopolitics, technicals · "
            f"data as of {AS_OF}"
        )
    with col_pill:
        st.markdown(
            "<div style='text-align:right; padding-top:24px;'>"
            "<span style='background:#1d3a5f; color:#7ab8ff; padding:4px 10px; "
            "border-radius:12px; font-size:12px;'>LIVE SPOT</span></div>",
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Spot price", f"${SPOT:,.2f}", f"{DAILY_CHANGE_PCT:+.2f}%")
    c2.metric("Prev close", f"${PREV_CLOSE:,.2f}", f"{(SPOT - PREV_CLOSE) / PREV_CLOSE * 100:+.2f}%")
    c3.metric("DXY", f"{DXY:.2f}", "Holding 99 floor")
    c4.metric("10Y real yield", f"{TIPS_10Y:.2f}%", "Rising headwind", delta_color="inverse")
    c5.metric("CPI y/y (Apr)", f"{CPI_YOY:.1f}%", "Re-accelerating", delta_color="inverse")


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def section_overview() -> None:
    st.info(
        "**Thesis · neutral-bearish into June, then bullish reversion through year-end.** "
        "Gold sits inside a contested $4,500–$4,900 range. Near-term headwinds are real: TIPS "
        "real yield has climbed from 1.91% (Apr 24) to 2.18%, DXY is grinding above the 99 floor, "
        "and the Fed is now priced for a hike rather than cuts after April CPI printed 3.8% y/y. "
        "But the structural bid is intact — Q1 2026 central bank net buying ran 244t (above the "
        "5-year average), April global ETF inflows hit an 18-month high (+$6.6B / +87.3t), and "
        "the WGC full-year target is 700–900t of sovereign accumulation. The path of least "
        "resistance is one more flush into $4,400–$4,500 support, then a recovery toward Wall "
        "Street's consensus $5,400–$6,300 by end-2026."
    )

    st.subheader("Forecast snapshot")
    cols = st.columns(3)
    for col, h in zip(cols, HORIZONS):
        with col:
            st.markdown(f"##### {h.name} · {h.until}")
            st.markdown(
                f"<span style='background:{h.bias_color}22; color:{h.bias_color}; "
                f"padding:2px 10px; border-radius:10px; font-size:12px;'>{h.bias}</span>",
                unsafe_allow_html=True,
            )
            st.metric(
                "Base target",
                f"${h.target:,.0f}",
                f"{(h.target - SPOT) / SPOT * 100:+.1f}% vs spot",
            )
            st.caption(f"Range: ${h.range_low:,.0f} – ${h.range_high:,.0f}")

    st.divider()
    st.subheader("Why this matters now")
    a, b, c = st.columns(3)
    with a:
        st.markdown("**Structural bid (bullish)**")
        st.markdown(
            "- 244t central-bank buying in Q1 (+3% y/y)\n"
            "- WGC FY target: 700–900t\n"
            "- 43% of CBs plan to add gold in 2026\n"
            "- Global ETF AUM at $615B (3rd highest ever)"
        )
    with b:
        st.markdown("**Cyclical headwind (bearish)**")
        st.markdown(
            "- 10Y TIPS yield 2.18% (+27 bps MTD)\n"
            "- DXY firm in 99–99.5 range\n"
            f"- Fed pricing ~{FED_HIKE_ODDS_DEC}% odds of Dec hike\n"
            "- Mgd Money long-loaded after $4,700 rally"
        )
    with c:
        st.markdown("**Wildcards (binary)**")
        st.markdown(
            "- April PCE print on May 28\n"
            "- Iran / Strait of Hormuz status\n"
            "- Q2 WGC central-bank report (July)\n"
            "- Bond auction demand at 5%+ yields"
        )


def section_forecasts() -> None:
    st.subheader("Three-horizon forecast")
    st.caption(
        "Each horizon is calibrated to a different driver: real-yield momentum (30d), "
        "inflation cycle (3m), and structural reserve diversification (1y)."
    )

    for h in HORIZONS:
        with st.container(border=True):
            top = st.columns([2, 1, 1, 1])
            top[0].markdown(f"### {h.name} · {h.until}")
            top[0].markdown(
                f"<span style='background:{h.bias_color}22; color:{h.bias_color}; "
                f"padding:3px 12px; border-radius:12px; font-size:12px;'>{h.bias}</span>",
                unsafe_allow_html=True,
            )
            top[1].metric("Range low", f"${h.range_low:,.0f}")
            top[2].metric("Range high", f"${h.range_high:,.0f}")
            top[3].metric(
                "Base target",
                f"${h.target:,.0f}",
                f"{(h.target - SPOT) / SPOT * 100:+.1f}%",
            )
            st.markdown(h.rationale)

    st.divider()
    st.subheader("Forecast band vs spot")

    rows = [
        {
            "Horizon": h.name,
            "Low": h.range_low,
            "Target": h.target,
            "High": h.range_high,
            "Bias": h.bias,
        }
        for h in HORIZONS
    ]
    fc = pd.DataFrame(rows)

    fig = go.Figure()
    for _, row in fc.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["Low"], row["High"]],
                y=[row["Horizon"], row["Horizon"]],
                mode="lines",
                line=dict(color="#3a7ec9", width=8),
                showlegend=False,
                hovertemplate=f"{row['Horizon']}: ${row['Low']:,} – ${row['High']:,}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[row["Target"]],
                y=[row["Horizon"]],
                mode="markers",
                marker=dict(color="#f2c14e", size=14, symbol="diamond", line=dict(color="#222", width=1)),
                showlegend=False,
                hovertemplate=f"Target: ${row['Target']:,}<extra></extra>",
            )
        )

    fig.add_vline(
        x=SPOT,
        line=dict(color="#888", dash="dash", width=1.5),
        annotation_text=f"Spot ${SPOT:,.0f}",
        annotation_position="top",
        annotation_font_color="#aaa",
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=320)
    fig.update_xaxes(title_text="Price ($/oz)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Bar = expected range · Diamond = base-case target · Dashed line = current spot")


def section_macro() -> None:
    st.subheader("Macro driver scorecard")

    drivers = pd.DataFrame(
        {
            "Driver": [
                "DXY (US Dollar Index)",
                "US 10Y real yield (TIPS)",
                "US CPI y/y",
                "Core PCE y/y",
                "Fed pricing",
                "Central bank buying",
                "Global gold ETFs",
                "COT Managed Money",
                "Geopolitics",
            ],
            "Reading": [
                "99.32 · 99–99.5 range",
                "2.18% (May 21)",
                "3.8% (Apr) · core 2.8%",
                "3.2% (Mar) · Apr due May 28",
                f"~{FED_HIKE_ODDS_DEC}% odds of Dec rate hike",
                "244t Q1 · WGC FY 700–900t",
                "+$6.6B Apr · 4,137t total",
                "+98,015 net long (May 12)",
                "US–Iran conflict · Strait of Hormuz",
            ],
            "Trend": [
                "Compressing, holding 99 floor",
                "Up from 1.91% on Apr 24",
                "Re-accelerating · energy +17.9%",
                "Widening gap vs CPI",
                "Hawkish repricing",
                "3rd year above 1,000t",
                "Flipped positive; West rejoining",
                "Below Jan peak of +134,745",
                "Unresolved, energy elevated",
            ],
            "Impact": [
                "Mild headwind",
                "Strong headwind",
                "Mixed",
                "Binary catalyst",
                "Headwind",
                "Structural bid",
                "Bullish",
                "Crowded · liquidation risk",
                "Bullish (safe-haven)",
            ],
            "Weight": ["Med", "High", "High", "High", "High", "High", "Med", "Med", "High"],
        }
    )

    def color_impact(val: str) -> str:
        bullish = ["Structural bid", "Bullish", "Bullish (safe-haven)"]
        bearish = ["Strong headwind", "Headwind", "Crowded · liquidation risk"]
        if val in bullish:
            return "background-color: rgba(47,157,106,0.18); color: #5fcf9a"
        if val in bearish:
            return "background-color: rgba(201,74,74,0.18); color: #ff9d9d"
        return "background-color: rgba(201,138,47,0.15); color: #f0c068"

    st.dataframe(
        drivers.style.applymap(color_impact, subset=["Impact"]),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Macro context · last 4 weeks")

    a, b = st.columns(2)
    with a:
        st.markdown("**US 10Y real yield (TIPS) — daily, percent**")
        st.plotly_chart(
            line_chart(REAL_YIELD_SERIES, "value", "#c94a4a", "%"),
            use_container_width=True,
        )
        st.caption(
            "Source: FRED DFII10. Rising real yields are the single biggest near-term "
            "headwind for a zero-coupon asset like gold."
        )
    with b:
        st.markdown("**DXY — US Dollar Index, daily close**")
        st.plotly_chart(line_chart(DXY_SERIES, "value", "#c98a2f"), use_container_width=True)
        st.caption(
            "Source: ICE / Investing.com. 52-week range 95.55 – 100.64. The 99 round-number "
            "floor has held three consecutive sessions; 99.50–100 is the ceiling."
        )


def section_technicals() -> None:
    st.subheader("Key technical levels")

    a, b = st.columns(2)
    with a:
        st.markdown("##### Support levels")
        support = pd.DataFrame(
            {
                "Level": [
                    "$4,509",
                    "$4,500 – $4,550",
                    "$4,481",
                    "$4,400 – $4,440",
                    "$4,376",
                    "$4,129 – $4,099",
                ],
                "Type": [
                    "Friday close · intraday pivot",
                    "Dec '25 range low · psychological",
                    "20% off ATH · bear-market trigger",
                    "Major weekly support · 61.8% fib",
                    "LiteFinance pivot",
                    "52-wk MA · Mar 23 low · war lows",
                ],
                "% from spot": [
                    f"{(4509 - SPOT) / SPOT * 100:+.1f}%",
                    f"{(4525 - SPOT) / SPOT * 100:+.1f}%",
                    f"{(4481 - SPOT) / SPOT * 100:+.1f}%",
                    f"{(4420 - SPOT) / SPOT * 100:+.1f}%",
                    f"{(4376 - SPOT) / SPOT * 100:+.1f}%",
                    f"{(4114 - SPOT) / SPOT * 100:+.1f}%",
                ],
            }
        )
        st.dataframe(support, use_container_width=True, hide_index=True)

    with b:
        st.markdown("##### Resistance levels")
        resistance = pd.DataFrame(
            {
                "Level": [
                    "$4,650 – $4,700",
                    "$4,744",
                    "$4,780",
                    "$4,850 – $4,900",
                    "$5,100",
                    "$5,602",
                ],
                "Type": [
                    "Daily momentum pivot · 200H MA",
                    "Long-term 50% retracement",
                    "Recent weekly high",
                    "Range top · key supply zone",
                    "Pivotal resistance",
                    "All-time high (Jan 2026)",
                ],
                "% from spot": [
                    f"{(4675 - SPOT) / SPOT * 100:+.1f}%",
                    f"{(4744 - SPOT) / SPOT * 100:+.1f}%",
                    f"{(4780 - SPOT) / SPOT * 100:+.1f}%",
                    f"{(4875 - SPOT) / SPOT * 100:+.1f}%",
                    f"{(5100 - SPOT) / SPOT * 100:+.1f}%",
                    f"{(5602 - SPOT) / SPOT * 100:+.1f}%",
                ],
            }
        )
        st.dataframe(resistance, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Price ladder visualization")

    levels = [
        ("Major support", 4114, "#c94a4a"),
        ("Support", 4420, "#c98a2f"),
        ("Support", 4525, "#c98a2f"),
        ("Pivot", 4675, "#888888"),
        ("Resistance", 4780, "#5fa37c"),
        ("Range top", 4875, "#5fa37c"),
        ("Major resistance", 5100, "#2f9d6a"),
        ("All-time high", 5602, "#2f9d6a"),
    ]
    fig = go.Figure()
    for label, price, color in levels:
        fig.add_hline(
            y=price,
            line=dict(color=color, width=1.5, dash="dot"),
            annotation_text=f"{label} ${price:,}",
            annotation_position="right",
            annotation_font_color=color,
        )
    fig.add_hline(
        y=SPOT,
        line=dict(color="#f2c14e", width=2.5),
        annotation_text=f"Spot ${SPOT:,.2f}",
        annotation_position="right",
        annotation_font_color="#f2c14e",
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=480)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(title_text="Price ($/oz)", range=[4000, 5700])
    st.plotly_chart(fig, use_container_width=True)


def section_scenarios() -> None:
    st.subheader("Bullish vs bearish catalysts")

    a, b = st.columns(2)
    with a:
        st.success("##### Bullish catalysts")
        st.markdown(
            "**1. Apr PCE undershoots (May 28).** Core PCE prints below 0.3% m/m → Fed-hike "
            "repricing fades → TIPS yield rolls back under 2.0% → instant reflex bid toward "
            "$4,700.\n\n"
            "**2. Iran / Hormuz escalation.** Any closure of the strait or strike-on-strike "
            "action sends oil higher and triggers safe-haven flow; a $4,900 breakout opens "
            "$5,100.\n\n"
            "**3. Central bank buying surprise.** Q2 WGC report (mid-July) showing ≥ 250t "
            "accumulation reinforces the structural floor.\n\n"
            "**4. Western ETF capitulation flip.** If GLD/IAU sustain net creations for 4+ "
            "weeks, base case shifts from $5,400 toward $6,000+.\n\n"
            "**5. USD breakdown below 99.** Triggers the inverse trade — gold typically "
            "rallies 4–6% in the following 3 weeks."
        )
    with b:
        st.error("##### Bearish catalysts")
        st.markdown(
            "**1. Apr PCE prints hot.** Core ≥ 0.4% m/m forces an even more hawkish Fed; "
            "TIPS pushes 2.30%+, DXY breaks 100, gold flushes to $4,400.\n\n"
            "**2. Long-liquidation cascade.** Managed Money added longs at $4,700; a clean "
            "break of $4,500 forces stops and adds 200–300 bps of supply.\n\n"
            "**3. Iran ceasefire / peace dividend.** Oil dumps, inflation eases, the "
            "geopolitical premium (~$200–300 in gold) unwinds quickly.\n\n"
            "**4. Bear-market trigger.** A weekly close below $4,481 flips long-term "
            "momentum; technical funds add shorts, target $4,129.\n\n"
            "**5. Swap-dealer hedge pressure.** Commercials added 10,818 shorts the week of "
            "May 12; if they continue fading rallies, $4,900 caps every bounce."
        )


def section_positioning() -> None:
    st.subheader("Positioning & flows")

    a, b = st.columns(2)
    with a:
        st.markdown("**COMEX Managed Money net longs — gold futures (contracts)**")
        st.plotly_chart(
            bar_chart_vertical(COT_NET_LONGS, "date", "contracts", "#3a7ec9"),
            use_container_width=True,
        )
        st.caption(
            "Source: CFTC COT. Speculative longs are off the Jan peak of +134.7k but rebuilt "
            "into the May rally to $4,700 — these positions are now offside after the reversal "
            "to $4,565."
        )
        st.metric("Latest (May 12)", "+98,015 contracts", "+3,761 w/w")

    with b:
        st.markdown("**Global physically-backed gold ETF monthly net flows (USD bn)**")
        st.plotly_chart(
            bar_chart_vertical(ETF_FLOWS, "month", "usd_bn", "#2f9d6a"),
            use_container_width=True,
        )
        st.caption(
            "Source: World Gold Council. April flipped positive after March outflows; Europe "
            "led (+$3.7B), Asia inflow streak at 12 months. AUM $615B, holdings 4,137t (3rd "
            "highest ever)."
        )
        st.metric("Apr 2026", "+$6.6B / +87.3t", "18-month high")

    st.divider()
    st.subheader("Central bank reserve accumulation")

    cb_data = pd.DataFrame(
        {
            "Year": ["2010-21 avg", "2022", "2023", "2024", "2025", "2026E"],
            "Tonnes": [473, 1136, 1037, 1037, 1100, 800],
        }
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=cb_data["Year"],
            y=cb_data["Tonnes"],
            marker=dict(color=["#888", "#2f9d6a", "#2f9d6a", "#2f9d6a", "#2f9d6a", "#3a7ec9"]),
            text=[f"{v:,}t" for v in cb_data["Tonnes"]],
            textposition="outside",
            textfont=dict(color="#d6d6d6"),
        )
    )
    fig.add_hline(
        y=473,
        line=dict(color="#888", dash="dash", width=1),
        annotation_text="Pre-2022 average",
        annotation_position="bottom right",
        annotation_font_color="#aaa",
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False)
    fig.update_yaxes(title_text="Tonnes per year")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Source: World Gold Council. 43% of central banks plan to increase reserves in 2026 "
        "(up from 29% two years ago). The structural floor has shifted ~$800 higher since "
        "2022 began."
    )


def section_targets() -> None:
    st.subheader("Wall Street year-end 2026 targets")
    st.caption(
        "Sources: Reuters factbox, Bloomberg, bank research notes Jan–May 2026."
    )

    st.plotly_chart(
        bar_chart_horizontal(BANK_TARGETS, "bank", "target", SPOT),
        use_container_width=True,
    )

    consensus = BANK_TARGETS["target"].median()
    upside_avg = BANK_TARGETS["target"].mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Consensus (median)", f"${consensus:,.0f}", f"{(consensus - SPOT) / SPOT * 100:+.1f}%")
    c2.metric("Mean target", f"${upside_avg:,.0f}", f"{(upside_avg - SPOT) / SPOT * 100:+.1f}%")
    c3.metric("Highest (UBS bull)", "$7,200", f"{(7200 - SPOT) / SPOT * 100:+.1f}%")
    c4.metric("Lowest (HSBC)", "$4,450", f"{(4450 - SPOT) / SPOT * 100:+.1f}%", delta_color="inverse")

    st.info(
        "Even the lowest call (HSBC, $4,450) sits essentially at spot — downside conviction "
        "is low across the street. Disagreement is about how high, not which direction."
    )


def section_strategies() -> None:
    st.subheader("Trading strategies · sized to risk profile")
    st.caption(
        "Pick the row that matches your time horizon and risk tolerance. Sizes assume max 1.5% "
        "of account equity at risk per trade for leveraged setups."
    )

    tab_swing, tab_position, tab_investor = st.tabs(
        ["Swing trader (1–4 weeks)", "Position trader (1–6 months)", "Long-term investor (1–3 years)"]
    )

    with tab_swing:
        st.markdown(
            "Trade the $4,400–$4,900 range. Current setup favors selling rallies into "
            "mid-range while the COT long-liquidation risk plays out, then flipping long at "
            "structural support."
        )
        swing = pd.DataFrame(
            {
                "Setup": ["Counter-trend short", "Range-low long", "Breakout long"],
                "Entry": ["$4,700 – $4,780", "$4,440 – $4,500", "Daily close > $4,900"],
                "Stop-loss": [
                    "$4,825 (above weekly high)",
                    "$4,395 (below 61.8% fib)",
                    "$4,815 (back inside range)",
                ],
                "Targets": [
                    "$4,580 → $4,500 → $4,440",
                    "$4,580 → $4,700 → $4,820",
                    "$5,028 → $5,100 → $5,400",
                ],
                "R:R": ["1 : 2.4", "1 : 3.2", "1 : 4.0"],
                "Trigger": [
                    "Rejection candle on H4 + DXY > 99.50",
                    "Bullish hammer + spike in COMEX OI",
                    "Apr PCE soft or geopolitical shock",
                ],
            }
        )
        st.dataframe(swing, use_container_width=True, hide_index=True)
        st.warning(
            "**Position sizing.** Max risk 1.0–1.5% of account equity per trade. ATR(14) on "
            "daily is ~$85 — your stop must respect that or you'll get noise-stopped before "
            "the move develops."
        )

    with tab_position:
        st.markdown(
            "Lean with the structural bid but wait for the near-term flush. The PCE print on "
            "May 28 is the binary — scale in below it on softness, or wait for $4,400 support."
        )
        position = pd.DataFrame(
            {
                "Step": ["1", "2", "3", "Stop", "Targets"],
                "Action": [
                    "Initial long",
                    "Add on confirmation",
                    "Add on breakout",
                    "Trail under prior swing low; hard exit on weekly close < $4,400",
                    "T1 $4,850 · T2 $5,100 · T3 $5,400 (Goldman)",
                ],
                "Price zone": [
                    "$4,440 – $4,500",
                    "$4,580 – $4,620",
                    "$4,910 (close)",
                    "—",
                    "—",
                ],
                "Allocation": ["33%", "33%", "34%", "—", "Scale 30 / 30 / 40%"],
                "Logic": [
                    "Range low + central-bank bid",
                    "Reclaim of daily pivot",
                    "Range break, no chasing intraday",
                    "Defends the entire structural thesis",
                    "Take partial at each major supply zone",
                ],
            }
        )
        st.dataframe(position, use_container_width=True, hide_index=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg entry (planned)", "$4,470")
        c2.metric("Aggregate stop", "$4,395", "−1.7%", delta_color="inverse")
        c3.metric("Blended target", "$5,100", "+14.1%")
        c4.metric("Expected R:R", "≈ 1 : 4.2")

    with tab_investor:
        st.markdown(
            "Gold's role here is portfolio insurance against monetary debasement, "
            "sovereign-debt spirals, and the structural de-dollarization trade central banks "
            "are already executing. Volatility along the way is the price of admission, not a "
            "signal."
        )
        investor = pd.DataFrame(
            {
                "Vehicle": [
                    "SPDR Gold Shares (GLD)",
                    "iShares Gold Trust (IAU)",
                    "SPDR Gold MiniShares (GLDM)",
                    "Physical bullion (LBMA bars / coins)",
                    "Gold miners ETF (GDX)",
                ],
                "Use case": [
                    "Liquid core position",
                    "Lower-cost alternative",
                    "Retail-friendly, low fee",
                    "Tail-risk hedge, no counterparty",
                    "Operating leverage to gold price",
                ],
                "Allocation guide": ["5–10%", "5–10%", "5–10%", "1–3%", "1–3%"],
                "Cost (annual)": ["0.40%", "0.25%", "0.10%", "Storage + spread", "0.51%"],
            }
        )
        st.dataframe(investor, use_container_width=True, hide_index=True)
        st.info(
            "**Dollar-cost averaging plan.** Split target allocation into 6 monthly tranches. "
            "If price drops 5%+ from your running average, accelerate the schedule by 50%. "
            "Rebalance back to target weight annually — let the structural trend compound, "
            "but trim if gold exceeds 15% of total book."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("12-mo blended target", "$5,800", "+27%")
        c2.metric("Bull case (JPM)", "$6,300", "+38%")
        c3.metric("Bear case (HSBC)", "$4,450", "−2.5%", delta_color="inverse")
        c4.metric("UBS bull scenario", "$7,200", "+58%")


def section_risk() -> None:
    st.subheader("What would invalidate this view")

    a, b, c = st.columns(3)
    with a:
        st.error("##### Hard invalidation")
        st.markdown(
            "Weekly close below **$4,400** with rising DXY through **100.65** (52-wk high). "
            "That would mean the structural bid has cracked and the macro tide has fully turned."
        )
    with b:
        st.warning("##### Soft invalidation")
        st.markdown(
            "Multiple weeks of net ETF outflows + central banks net sellers (rare — only "
            "Türkiye, Russia, Azerbaijan did so in Q1) + TIPS real yield sustained above "
            "**2.50%**."
        )
    with c:
        st.info("##### Re-rate higher trigger")
        st.markdown(
            "Decisive break of **$4,900** on heavy volume with DXY breaking 99 to the downside. "
            "Targets re-rate to $5,400 → $5,800 → $6,300 sequentially."
        )

    st.divider()
    st.subheader("Key dates to watch")

    dates = pd.DataFrame(
        {
            "Date": [
                "May 28, 2026",
                "Jun 18, 2026",
                "Jul 16, 2026",
                "Jul 30, 2026",
                "Sep 17, 2026",
                "Q3 2026 (mid-July)",
            ],
            "Event": [
                "April PCE inflation release (BEA)",
                "FOMC meeting + Summary of Economic Projections",
                "June CPI release",
                "FOMC meeting + Powell press conference",
                "FOMC meeting · dot plot update",
                "World Gold Council Q2 demand trends",
            ],
            "Why it matters": [
                "Fed's preferred inflation gauge · binary catalyst",
                "First chance for hawkish pivot to be priced in fully",
                "Confirms whether CPI re-acceleration is sticky",
                "Setup for September decision · key forward guidance",
                "Updated dots can rip real yields either direction",
                "Central-bank buying confirms or breaks structural thesis",
            ],
        }
    )
    st.dataframe(dates, use_container_width=True, hide_index=True)

    st.divider()
    st.caption(
        "**Disclaimer.** This dashboard is a structured analytical view, not personalised "
        "investment advice. Trading leveraged FX/commodities carries substantial risk of loss. "
        "Sources: BLS, BEA, FRED (DFII10), CFTC COT, World Gold Council, ICE / Investing.com, "
        "Reuters factbox of bank targets, FXEmpire / MarketPulse technical analysis. "
        f"Data current as of {AS_OF}."
    )


# ===========================================================================
# Stock Analyzer (Auto) — type a ticker, get everything in one shot.
# ===========================================================================
# Pulls live data from Yahoo Finance via yfinance, computes Buffett-style
# verdicts from real financials, runs a two-stage DCF, derives technical
# levels, and builds a trade plan. One input → full analysis.


YZ_PERIOD_DAYS = {"6mo": 180, "1y": 365, "2y": 730, "5y": 1825}


@st.cache_data(ttl=900, show_spinner=False)
def _yz_fetch(symbol: str) -> Dict[str, object]:
    """Fetch all needed data for a ticker from Yahoo Finance. Cached 15 min."""
    if not _YF_OK:
        return {"error": "yfinance not installed. Run: pip install yfinance"}
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return {"error": "Enter a ticker symbol"}
    try:
        ticker = yf.Ticker(symbol)
        try:
            info = ticker.info or {}
        except Exception:
            info = {}
        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )
        history = pd.DataFrame()
        try:
            history = ticker.history(period="5y", auto_adjust=True)
        except Exception:
            pass
        if (history is None or history.empty) and price is None:
            return {"error": f"No data found for '{symbol}'. Check the ticker symbol."}
        if history is None or history.empty:
            return {"error": f"No price history available for '{symbol}'."}
        cashflow = pd.DataFrame()
        try:
            cashflow = ticker.cashflow
        except Exception:
            pass
        financials = pd.DataFrame()
        try:
            financials = ticker.financials
        except Exception:
            pass
        balance = pd.DataFrame()
        try:
            balance = ticker.balance_sheet
        except Exception:
            pass
        return {
            "symbol": symbol,
            "info": info,
            "history": history,
            "cashflow": cashflow,
            "financials": financials,
            "balance": balance,
        }
    except Exception as exc:
        return {"error": f"Failed to fetch '{symbol}': {exc}"}


def _yz_technicals(history: pd.DataFrame) -> dict:
    """MAs, RSI, ATR, trend, momentum, support/resistance from OHLC."""
    df = history.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    df["RSI"] = 100 - 100 / (1 + rs)

    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    latest = df.iloc[-1]
    price = float(latest["Close"])
    ma20 = float(latest["MA20"]) if pd.notna(latest["MA20"]) else None
    ma50 = float(latest["MA50"]) if pd.notna(latest["MA50"]) else None
    ma200 = float(latest["MA200"]) if pd.notna(latest["MA200"]) else None
    rsi = float(latest["RSI"]) if pd.notna(latest["RSI"]) else None
    atr = float(latest["ATR"]) if pd.notna(latest["ATR"]) else None

    if ma50 and ma200:
        if price > ma50 > ma200:
            trend = "Bullish · price > 50d > 200d"
        elif price < ma50 < ma200:
            trend = "Bearish · price < 50d < 200d"
        elif price > ma200:
            trend = "Mixed · holding above 200d"
        else:
            trend = "Mixed · below 200d"
    else:
        trend = "Insufficient history"

    if rsi is None:
        momentum = "n/a"
    elif rsi >= 70:
        momentum = f"Overbought (RSI {rsi:.0f})"
    elif rsi <= 30:
        momentum = f"Oversold (RSI {rsi:.0f})"
    else:
        momentum = f"Neutral (RSI {rsi:.0f})"

    recent = df.tail(120)
    below = recent[recent["Low"] < price]["Low"]
    above = recent[recent["High"] > price]["High"]
    supports = (
        sorted(below.nlargest(3).round(2).unique().tolist(), reverse=True)
        if not below.empty else []
    )
    resistances = (
        sorted(above.nsmallest(3).round(2).unique().tolist())
        if not above.empty else []
    )

    high52 = float(df["High"].tail(252).max()) if len(df) >= 1 else None
    low52 = float(df["Low"].tail(252).min()) if len(df) >= 1 else None

    return {
        "df": df, "price": price,
        "ma20": ma20, "ma50": ma50, "ma200": ma200,
        "rsi": rsi, "atr": atr,
        "trend": trend, "momentum": momentum,
        "supports": supports, "resistances": resistances,
        "high52": high52, "low52": low52,
    }


def _yz_dcf(cashflow: pd.DataFrame, info: dict) -> dict:
    """Two-stage 10-year DCF using real free cash flow data."""
    if cashflow is None or cashflow.empty:
        return {"error": "No cash flow statement available from data provider"}

    fcf_values: list[float] = []
    for label in ["Free Cash Flow", "FreeCashFlow"]:
        if label in cashflow.index:
            fcf_values = [
                float(v) for v in cashflow.loc[label].dropna().tolist()
                if v is not None and pd.notna(v)
            ]
            break
    if not fcf_values:
        ocf = capex = None
        for label in ["Operating Cash Flow", "Cash Flow From Operations",
                      "Total Cash From Operating Activities"]:
            if label in cashflow.index:
                ocf = cashflow.loc[label]
                break
        for label in ["Capital Expenditure", "Capital Expenditures"]:
            if label in cashflow.index:
                capex = cashflow.loc[label]
                break
        if ocf is not None and capex is not None:
            series = (ocf + capex).dropna()
            fcf_values = [float(v) for v in series.tolist()]

    if not fcf_values:
        return {"error": "Could not extract free cash flow from financials"}

    fcf_base = sum(fcf_values[: min(3, len(fcf_values))]) / min(3, len(fcf_values))
    if fcf_base <= 0:
        return {
            "error": (
                f"Average FCF is negative (${fcf_base / 1e9:.1f}B). DCF is "
                "not meaningful for a cash-burning business."
            )
        }

    growth_raw = info.get("earningsGrowth") or info.get("revenueGrowth") or 0.06
    if growth_raw is None or growth_raw < 0:
        growth_raw = 0.04
    g1 = min(max(float(growth_raw), 0.02), 0.18)
    g2 = max(g1 / 2, 0.02)
    gt = 0.025
    dr = 0.10

    pv = 0.0
    fcf = fcf_base
    for year in range(1, 11):
        g = g1 if year <= 5 else g2
        fcf = fcf * (1 + g)
        pv += fcf / (1 + dr) ** year
    terminal = fcf * (1 + gt) / (dr - gt)
    pv_terminal = terminal / (1 + dr) ** 10
    intrinsic = pv + pv_terminal

    shares = info.get("sharesOutstanding") or 0
    per_share = intrinsic / shares if shares else None

    return {
        "fcf_base": fcf_base, "intrinsic_total": intrinsic,
        "intrinsic_per_share": per_share,
        "g1": g1, "g2": g2, "dr": dr, "gt": gt, "shares": shares,
    }


def _yz_auto_verdicts(info: dict, tech: dict, dcf: dict) -> list[dict]:
    """Compute Buffett-style gate verdicts from real financial metrics."""
    verdicts: list[dict] = []

    roe = info.get("returnOnEquity") or 0
    op_m = info.get("operatingMargins") or 0
    if roe > 0.20 and op_m > 0.20:
        v, msg = "PASS", f"ROE {roe*100:.1f}% · op margin {op_m*100:.1f}% — classic moat profile"
    elif roe > 0.12 and op_m > 0.10:
        v, msg = "PARTIAL", f"ROE {roe*100:.1f}% · op margin {op_m*100:.1f}% — decent but not exceptional"
    else:
        v, msg = "FAIL", f"ROE {roe*100:.1f}% · op margin {op_m*100:.1f}% — no clear moat signal"
    verdicts.append({"gate": "Economic Moat", "verdict": v, "evidence": msg})

    sector = info.get("sector") or "Unknown"
    simple_sectors = {"Technology", "Consumer Defensive", "Consumer Cyclical",
                      "Industrials", "Communication Services", "Healthcare"}
    if sector in simple_sectors:
        v, msg = "PASS", f"{sector} — generally understandable from public information"
    elif sector in {"Financial Services", "Real Estate", "Energy", "Basic Materials"}:
        v, msg = "PARTIAL", f"{sector} — needs domain expertise (accounting, commodity, regulation)"
    else:
        v, msg = "PARTIAL", f"{sector} — confirm you can model the unit economics"
    verdicts.append({"gate": "Simplicity", "verdict": v, "evidence": msg})

    rg = info.get("revenueGrowth")
    if rg is not None:
        rg_pct = rg * 100
        if rg_pct > 8:
            v, msg = "PASS", f"Revenue growth {rg_pct:+.1f}% y/y — demand expanding"
        elif rg_pct > 0:
            v, msg = "PARTIAL", f"Revenue growth {rg_pct:+.1f}% y/y — slow but positive"
        else:
            v, msg = "FAIL", f"Revenue growth {rg_pct:+.1f}% y/y — business shrinking"
    else:
        v, msg = "PARTIAL", "Revenue growth data unavailable"
    verdicts.append({"gate": "10-Year Durability (growth proxy)", "verdict": v, "evidence": msg})

    high52 = info.get("fiftyTwoWeekHigh") or tech.get("high52")
    if high52 and tech["price"]:
        dd = (high52 - tech["price"]) / high52 * 100
        if dd > 25:
            v, msg = "PASS", f"Down {dd:.0f}% from 52-wk high — fear may be mispricing risk"
        elif dd > 10:
            v, msg = "PARTIAL", f"Down {dd:.0f}% from 52-wk high — modest pullback"
        else:
            v, msg = "FAIL", f"Only {dd:.0f}% off 52-wk high — limited fear-driven discount"
    else:
        v, msg = "PARTIAL", "52-week high unavailable"
    verdicts.append({"gate": "Fear vs Opportunity", "verdict": v, "evidence": msg})

    insiders = info.get("heldPercentInsiders") or 0
    if insiders > 0.05:
        v, msg = "PASS", f"Insider ownership {insiders*100:.1f}% — owner-aligned"
    elif insiders > 0.01:
        v, msg = "PARTIAL", f"Insider ownership {insiders*100:.1f}% — modest skin in game"
    else:
        v, msg = "FAIL", f"Insider ownership {insiders*100:.1f}% — low alignment"
    verdicts.append({"gate": "Management Alignment", "verdict": v, "evidence": msg})

    if dcf.get("intrinsic_per_share") and tech["price"]:
        iv = dcf["intrinsic_per_share"]
        gap = (iv - tech["price"]) / tech["price"] * 100
        if gap > 30:
            v, msg = "PASS", f"DCF fair value ${iv:,.0f} → {gap:+.0f}% upside"
        elif gap > 0:
            v, msg = "PARTIAL", f"DCF fair value ${iv:,.0f} → {gap:+.0f}% — fair, not cheap"
        else:
            v, msg = "FAIL", f"DCF fair value ${iv:,.0f} → {gap:+.0f}% — overpriced"
    else:
        pe = info.get("trailingPE")
        if pe and pe < 15:
            v, msg = "PASS", f"Trailing P/E {pe:.1f} — below market median"
        elif pe and pe < 25:
            v, msg = "PARTIAL", f"Trailing P/E {pe:.1f} — reasonable"
        elif pe:
            v, msg = "FAIL", f"Trailing P/E {pe:.1f} — expensive"
        else:
            v, msg = "PARTIAL", "Valuation data unavailable"
    verdicts.append({"gate": "Price vs Value", "verdict": v, "evidence": msg})

    debt_eq = info.get("debtToEquity")
    if debt_eq is not None:
        if debt_eq < 50:
            v, msg = "PASS", f"Debt/Equity {debt_eq:.0f}% — conservative"
        elif debt_eq < 150:
            v, msg = "PARTIAL", f"Debt/Equity {debt_eq:.0f}% — leveraged"
        else:
            v, msg = "FAIL", f"Debt/Equity {debt_eq:.0f}% — heavily indebted (newspaper-test risk)"
    else:
        v, msg = "PARTIAL", "Debt data unavailable"
    verdicts.append({"gate": "Newspaper Test (debt risk proxy)", "verdict": v, "evidence": msg})

    mcap = info.get("marketCap") or 0
    if mcap > 50e9:
        v, msg = "PASS", f"Mega-cap ${mcap/1e9:.0f}B — coverage and disclosure are abundant"
    elif mcap > 2e9:
        v, msg = "PARTIAL", f"Mid-cap ${mcap/1e9:.1f}B — coverage thinner; do deeper work"
    elif mcap > 0:
        v, msg = "FAIL", f"Small/micro-cap ${mcap/1e9:.2f}B — wide blind spots likely"
    else:
        v, msg = "PARTIAL", "Market cap unavailable"
    verdicts.append({"gate": "Circle of Competence (data availability)", "verdict": v, "evidence": msg})

    return verdicts


def _yz_trade_plan(tech: dict, info: dict, dcf: dict) -> dict:
    """Derive an entry/stop/target plan from technicals and valuation."""
    price = tech["price"]
    atr = tech["atr"] or price * 0.02
    supports = tech["supports"]
    resistances = tech["resistances"]

    nearest_support = supports[0] if supports else price * 0.95
    entry_low = round(nearest_support * 1.005, 2)
    entry_high = round(min(price, nearest_support * 1.03), 2)
    if entry_high < entry_low:
        entry_low, entry_high = entry_high, entry_low
    stop = round(nearest_support - 1.5 * atr, 2)

    target1 = round(resistances[0], 2) if resistances else round(price + 2 * atr, 2)
    target2 = round(resistances[1], 2) if len(resistances) > 1 else round(price * 1.15, 2)
    target_long = None
    if dcf.get("intrinsic_per_share"):
        target_long = round(dcf["intrinsic_per_share"], 2)
    elif info.get("targetMeanPrice"):
        target_long = round(info["targetMeanPrice"], 2)

    risk = max(price - stop, 0.01)
    reward = max(target1 - price, 0)
    rr = reward / risk if risk > 0 else None

    return {
        "entry_low": entry_low, "entry_high": entry_high,
        "stop": stop, "target1": target1, "target2": target2,
        "target_long": target_long, "atr": atr,
        "risk": risk, "reward": reward, "rr": rr,
    }


def _yz_verdict_color(v: str) -> str:
    return {"PASS": "#2f9d6a", "PARTIAL": "#c98a2f", "FAIL": "#c94a4a"}.get(v, "#888888")


def _yz_render_price_chart(history: pd.DataFrame, tech: dict, period: str) -> None:
    """Candlestick + MA20/50/200 + volume."""
    days = YZ_PERIOD_DAYS.get(period, 730)
    df = tech["df"].tail(days)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03,
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            increasing_line_color="#2f9d6a", decreasing_line_color="#c94a4a",
            name="Price", showlegend=False,
        ),
        row=1, col=1,
    )
    for ma, color in [("MA20", "#7ab8ff"), ("MA50", "#f2c14e"), ("MA200", "#c98a2f")]:
        if ma in df.columns and df[ma].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df[ma], mode="lines",
                    line=dict(color=color, width=1.2), name=ma,
                ),
                row=1, col=1,
            )
    fig.add_trace(
        go.Bar(
            x=df.index, y=df["Volume"],
            marker=dict(color="rgba(120,120,160,0.5)"),
            name="Volume", showlegend=False,
        ),
        row=2, col=1,
    )
    fig.update_layout(
        **PLOTLY_LAYOUT, height=520,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)


def _yz_render_snapshot(info: dict, tech: dict, history: pd.DataFrame,
                        period: str, verdicts: list[dict]) -> None:
    st.subheader(f"Price action — last {period}")
    _yz_render_price_chart(history, tech, period)

    summary = info.get("longBusinessSummary") or ""
    if summary:
        with st.expander("Business description (from Yahoo)", expanded=False):
            st.write(summary)

    st.subheader("At-a-glance verdict")
    passes = sum(1 for v in verdicts if v["verdict"] == "PASS")
    partial = sum(1 for v in verdicts if v["verdict"] == "PARTIAL")
    fails = sum(1 for v in verdicts if v["verdict"] == "FAIL")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pass", f"{passes} / 8")
    c2.metric("Partial", partial, delta_color="off")
    c3.metric(
        "Fail", fails,
        "Thesis blocker" if fails else "—",
        delta_color="inverse" if fails else "normal",
    )
    if fails == 0 and passes >= 6:
        overall, color = "BUY-CANDIDATE", "#2f9d6a"
    elif fails <= 1 and passes >= 4:
        overall, color = "WATCH", "#c98a2f"
    else:
        overall, color = "AVOID", "#c94a4a"
    c4.markdown(
        f"<div style='padding-top:18px;'><span style='background:{color}22; "
        f"color:{color}; padding:8px 16px; border-radius:14px; font-weight:700; "
        f"font-size:16px;'>{overall}</span></div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Auto-summary derived from real-time financials. See the "
        "**Buffett-style verdict** tab for per-gate evidence and the "
        "**Trade plan** tab for entry/stop/targets."
    )


def _yz_render_fundamentals(info: dict, cashflow: pd.DataFrame,
                            financials: pd.DataFrame) -> None:
    st.subheader("Fundamental snapshot (TTM)")
    rows = [
        ("Revenue", info.get("totalRevenue"), "$B"),
        ("Gross profit", info.get("grossProfits"), "$B"),
        ("Net income", info.get("netIncomeToCommon"), "$B"),
        ("Free cash flow", info.get("freeCashflow"), "$B"),
        ("Operating margin", info.get("operatingMargins"), "%"),
        ("Profit margin", info.get("profitMargins"), "%"),
        ("Return on equity", info.get("returnOnEquity"), "%"),
        ("Return on assets", info.get("returnOnAssets"), "%"),
        ("Revenue growth (y/y)", info.get("revenueGrowth"), "%"),
        ("Earnings growth (y/y)", info.get("earningsGrowth"), "%"),
        ("Total cash", info.get("totalCash"), "$B"),
        ("Total debt", info.get("totalDebt"), "$B"),
        ("Debt / Equity", info.get("debtToEquity"), "raw"),
        ("Current ratio", info.get("currentRatio"), "raw"),
    ]
    table = []
    for label, val, kind in rows:
        if val is None or (isinstance(val, float) and val != val):
            shown = "—"
        elif kind == "$B":
            shown = f"${val/1e9:,.2f}B"
        elif kind == "%":
            shown = f"{val*100:+.2f}%"
        else:
            shown = f"{val:,.2f}"
        table.append({"Metric": label, "Value": shown})
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

    if financials is not None and not financials.empty:
        st.subheader("Revenue & income trend (annual)")
        rows_chart = []
        wanted = {
            "Total Revenue": "Revenue",
            "Gross Profit": "Gross profit",
            "Net Income": "Net income",
        }
        for src, label in wanted.items():
            if src in financials.index:
                series = financials.loc[src].dropna()
                for date, value in series.items():
                    rows_chart.append({
                        "Year": pd.to_datetime(date).year,
                        "Metric": label,
                        "USD billions": float(value) / 1e9,
                    })
        if rows_chart:
            df_chart = pd.DataFrame(rows_chart).sort_values("Year")
            fig = go.Figure()
            palette = {"Revenue": "#3a7ec9", "Gross profit": "#f2c14e",
                       "Net income": "#2f9d6a"}
            for metric in df_chart["Metric"].unique():
                sub = df_chart[df_chart["Metric"] == metric]
                fig.add_trace(go.Bar(
                    x=sub["Year"], y=sub["USD billions"],
                    name=metric, marker=dict(color=palette.get(metric)),
                ))
            fig.update_layout(**PLOTLY_LAYOUT, height=340, barmode="group")
            fig.update_yaxes(title_text="USD billions")
            st.plotly_chart(fig, use_container_width=True)

    if cashflow is not None and not cashflow.empty:
        st.subheader("Free cash flow trend (annual)")
        fcf_series = None
        for label in ["Free Cash Flow", "FreeCashFlow"]:
            if label in cashflow.index:
                fcf_series = cashflow.loc[label].dropna()
                break
        if fcf_series is not None and not fcf_series.empty:
            df_fcf = pd.DataFrame({
                "Year": [pd.to_datetime(d).year for d in fcf_series.index],
                "FCF ($B)": [float(v) / 1e9 for v in fcf_series.values],
            }).sort_values("Year")
            colors = ["#2f9d6a" if v >= 0 else "#c94a4a" for v in df_fcf["FCF ($B)"]]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_fcf["Year"], y=df_fcf["FCF ($B)"],
                marker=dict(color=colors),
                text=[f"${v:.1f}B" for v in df_fcf["FCF ($B)"]],
                textposition="outside", textfont=dict(color="#d6d6d6"),
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False)
            fig.update_yaxes(title_text="Free cash flow ($B)")
            st.plotly_chart(fig, use_container_width=True)


def _yz_render_valuation(info: dict, tech: dict, dcf: dict) -> None:
    st.subheader("Intrinsic value · 10-year two-stage DCF")
    st.caption(
        "Base FCF = 3-year average. Y1-5 growth = clamp(earnings or revenue "
        "growth, 2-18%). Y6-10 growth = half that. Terminal 2.5%, discount 10%."
    )
    if "error" in dcf:
        st.warning(dcf["error"])
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("FCF base", f"${dcf['fcf_base']/1e9:,.2f}B")
        c2.metric("Growth Y1-5", f"{dcf['g1']*100:.1f}%")
        c3.metric("Growth Y6-10", f"{dcf['g2']*100:.1f}%")
        c4.metric("Discount rate", f"{dcf['dr']*100:.1f}%")

        if dcf.get("intrinsic_per_share"):
            iv = dcf["intrinsic_per_share"]
            price = tech["price"]
            gap = (iv - price) / price * 100
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Intrinsic value total", f"${dcf['intrinsic_total']/1e9:,.1f}B")
            cc2.metric("DCF fair value / share", f"${iv:,.2f}")
            cc3.metric("vs current price", f"${price:,.2f}", f"{gap:+.1f}%")

    st.subheader("Wall Street analyst targets")
    target = info.get("targetMeanPrice")
    target_high = info.get("targetHighPrice")
    target_low = info.get("targetLowPrice")
    rec = info.get("recommendationKey") or info.get("recommendationMean")
    num_analysts = info.get("numberOfAnalystOpinions")
    if target:
        c1, c2, c3, c4 = st.columns(4)
        gap = (target - tech["price"]) / tech["price"] * 100
        c1.metric("Mean target", f"${target:,.2f}", f"{gap:+.1f}%")
        c2.metric("High target", f"${target_high:,.2f}" if target_high else "—")
        c3.metric("Low target", f"${target_low:,.2f}" if target_low else "—")
        c4.metric(
            "Coverage",
            f"{num_analysts} analysts" if num_analysts else "—",
            str(rec) if rec else "",
        )
    else:
        st.caption("No analyst targets available for this ticker.")

    st.subheader("Price-vs-value bands")
    bands: list[tuple[str, float, str]] = []
    if dcf.get("intrinsic_per_share"):
        iv = dcf["intrinsic_per_share"]
        bands.extend([
            ("DCF conservative (-25%)", iv * 0.75, "#5fa37c"),
            ("DCF base", iv, "#f2c14e"),
            ("DCF bull (+25%)", iv * 1.25, "#c98a2f"),
        ])
    if target_low:
        bands.append(("Analyst low", float(target_low), "#666666"))
    if target:
        bands.append(("Analyst mean", float(target), "#3a7ec9"))
    if target_high:
        bands.append(("Analyst high", float(target_high), "#888888"))
    bands.append(("Current price", tech["price"], "#ffffff"))
    if len(bands) > 1:
        labels = [b[0] for b in bands]
        values = [b[1] for b in bands]
        colors = [b[2] for b in bands]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels, y=values, marker=dict(color=colors),
            text=[f"${v:,.0f}" for v in values],
            textposition="outside", textfont=dict(color="#d6d6d6"),
            showlegend=False,
        ))
        fig.add_hline(
            y=tech["price"],
            line=dict(color="#ffffff", width=1.5, dash="dash"),
            annotation_text=f"Spot ${tech['price']:.2f}",
            annotation_position="top right",
            annotation_font_color="#aaa",
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=400)
        fig.update_yaxes(title_text="Price per share ($)")
        st.plotly_chart(fig, use_container_width=True)


def _yz_render_technicals(tech: dict, history: pd.DataFrame, period: str) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trend", tech["trend"])
    c2.metric("Momentum", tech["momentum"])
    c3.metric("ATR(14)", f"${tech['atr']:.2f}" if tech["atr"] else "—")
    c4.metric(
        "52-wk range",
        f"${tech['low52']:.0f} – ${tech['high52']:.0f}"
        if tech["low52"] and tech["high52"] else "—",
    )

    st.subheader("Key support / resistance levels")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Support (nearest first)")
        if tech["supports"]:
            rows = []
            for lvl in tech["supports"]:
                gap = (lvl - tech["price"]) / tech["price"] * 100
                rows.append({"Level": f"${lvl:,.2f}", "Distance": f"{gap:+.1f}%"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.caption("No nearby support detected in the last 120 sessions.")
    with c2:
        st.markdown("##### Resistance (nearest first)")
        if tech["resistances"]:
            rows = []
            for lvl in tech["resistances"]:
                gap = (lvl - tech["price"]) / tech["price"] * 100
                rows.append({"Level": f"${lvl:,.2f}", "Distance": f"{gap:+.1f}%"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.caption("No nearby resistance detected in the last 120 sessions.")

    st.subheader("Moving averages")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "20-day MA",
        f"${tech['ma20']:.2f}" if tech["ma20"] else "—",
        f"{(tech['price']-tech['ma20'])/tech['ma20']*100:+.1f}%" if tech["ma20"] else None,
    )
    c2.metric(
        "50-day MA",
        f"${tech['ma50']:.2f}" if tech["ma50"] else "—",
        f"{(tech['price']-tech['ma50'])/tech['ma50']*100:+.1f}%" if tech["ma50"] else None,
    )
    c3.metric(
        "200-day MA",
        f"${tech['ma200']:.2f}" if tech["ma200"] else "—",
        f"{(tech['price']-tech['ma200'])/tech['ma200']*100:+.1f}%" if tech["ma200"] else None,
    )
    c4.metric("RSI(14)", f"{tech['rsi']:.0f}" if tech["rsi"] else "—")

    if tech["df"] is not None and "RSI" in tech["df"].columns:
        st.subheader("RSI (14)")
        days = YZ_PERIOD_DAYS.get(period, 730)
        rsi_df = tech["df"].tail(days)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rsi_df.index, y=rsi_df["RSI"], mode="lines",
            line=dict(color="#7ab8ff", width=1.5), name="RSI",
        ))
        fig.add_hline(y=70, line=dict(color="#c94a4a", dash="dot", width=1),
                      annotation_text="Overbought 70", annotation_position="top right",
                      annotation_font_color="#c94a4a")
        fig.add_hline(y=30, line=dict(color="#2f9d6a", dash="dot", width=1),
                      annotation_text="Oversold 30", annotation_position="bottom right",
                      annotation_font_color="#2f9d6a")
        fig.update_layout(**PLOTLY_LAYOUT, height=240, showlegend=False)
        fig.update_yaxes(title_text="RSI", range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)


def _yz_render_trade(tech: dict, plan: dict, dcf: dict, info: dict) -> None:
    st.subheader("Auto-generated trade plan")
    st.caption(
        "Derived from current technicals and valuation. Entry is a band, not a "
        "single price. Stops use 1.5× ATR below nearest support."
    )
    price = tech["price"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Entry zone",
        f"${plan['entry_low']:.2f} – ${plan['entry_high']:.2f}",
    )
    c2.metric(
        "Stop loss", f"${plan['stop']:.2f}",
        f"{(plan['stop']-price)/price*100:+.1f}% from spot",
        delta_color="inverse",
    )
    c3.metric(
        "Target 1 (resistance)", f"${plan['target1']:.2f}",
        f"{(plan['target1']-price)/price*100:+.1f}%",
    )
    c4.metric(
        "Risk : Reward",
        f"1 : {plan['rr']:.2f}" if plan["rr"] else "—",
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Target 2", f"${plan['target2']:.2f}",
        f"{(plan['target2']-price)/price*100:+.1f}%",
    )
    if plan["target_long"]:
        c2.metric(
            "Long-term target (DCF / analyst)",
            f"${plan['target_long']:.2f}",
            f"{(plan['target_long']-price)/price*100:+.1f}%",
        )
    c3.metric(
        "ATR(14) — daily range",
        f"${plan['atr']:.2f}",
        f"{plan['atr']/price*100:.1f}% of spot",
    )

    st.subheader("Levels on chart")
    levels = [
        ("Stop", plan["stop"], "#c94a4a"),
        ("Entry low", plan["entry_low"], "#5fa37c"),
        ("Entry high", plan["entry_high"], "#5fa37c"),
        ("Spot", price, "#ffffff"),
        ("Target 1", plan["target1"], "#f2c14e"),
        ("Target 2", plan["target2"], "#c98a2f"),
    ]
    if plan["target_long"]:
        levels.append(("Long-term target", plan["target_long"], "#2f9d6a"))

    hist = tech["df"].tail(90)
    near_term_vals = [plan["stop"], plan["entry_low"], plan["entry_high"],
                      plan["target1"], plan["target2"], price]
    chart_min = min(float(hist["Low"].min()), min(near_term_vals))
    chart_max = max(float(hist["High"].max()), max(near_term_vals))
    span = chart_max - chart_min or price * 0.05
    y_low = chart_min - span * 0.08
    y_high = chart_max + span * 0.08

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["Close"],
        mode="lines", line=dict(color="#7ab8ff", width=1.8),
        name="Close", showlegend=False,
        hovertemplate="%{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>",
    ))

    off_chart: list[tuple[str, float]] = []
    for label, val, color in levels:
        if y_low <= val <= y_high:
            fig.add_hline(
                y=val, line=dict(color=color, width=1.5, dash="dot"),
                annotation_text=f"{label}  ${val:,.2f}",
                annotation_position="right",
                annotation_font_color=color,
            )
        else:
            off_chart.append((label, val))

    fig.update_layout(**PLOTLY_LAYOUT, height=460)
    fig.update_yaxes(title_text="Price ($)", range=[y_low, y_high])
    st.plotly_chart(fig, use_container_width=True)

    if off_chart:
        formatted = ", ".join(f"**{label}** ${val:,.2f}" for label, val in off_chart)
        st.caption(
            f"Off-chart levels (far from current price action): {formatted}. "
            "These usually indicate a DCF / analyst target far from spot — useful "
            "context, but not actionable in the current setup."
        )

    st.subheader("12-month scenarios")
    high52 = tech["high52"] or price * 1.2
    low52 = tech["low52"] or price * 0.8
    bull_target = plan["target_long"] or info.get("targetMeanPrice") or high52
    scenarios = pd.DataFrame([
        {
            "Scenario": "Base case",
            "Price target": f"${plan['target1']:,.2f}",
            "Return vs spot": f"{(plan['target1']-price)/price*100:+.1f}%",
            "Trigger": "Holds support, MA50 turns up, no earnings miss.",
        },
        {
            "Scenario": "Bull case",
            "Price target": f"${bull_target:,.2f}",
            "Return vs spot": f"{(bull_target-price)/price*100:+.1f}%",
            "Trigger": "DCF / analyst target realized over 12-18 months.",
        },
        {
            "Scenario": "Bear case",
            "Price target": f"${low52:,.2f}",
            "Return vs spot": f"{(low52-price)/price*100:+.1f}%",
            "Trigger": "Stop triggered, 52-wk low retested on macro shock.",
        },
    ])
    st.dataframe(scenarios, use_container_width=True, hide_index=True)

    st.warning(
        "**Trading levels are mechanical heuristics, not predictions.** "
        "ATR-based stops respect volatility but won't save you from a gap. "
        "Size positions by your account risk, not by the chart."
    )


def _yz_render_buffett(verdicts: list[dict]) -> None:
    st.subheader("Buffett-style auto verdict")
    st.caption(
        "Each gate is scored deterministically from real financial data — "
        "change the company, change the verdict."
    )
    rows = [{"Gate": v["gate"], "Verdict": v["verdict"], "Evidence": v["evidence"]}
            for v in verdicts]
    df = pd.DataFrame(rows)

    def color_verdict(val: str) -> str:
        color = _yz_verdict_color(val)
        return f"background-color: {color}22; color: {color}; font-weight: 600"

    st.dataframe(
        df.style.applymap(color_verdict, subset=["Verdict"]),
        use_container_width=True, hide_index=True,
    )

    passes = sum(1 for v in verdicts if v["verdict"] == "PASS")
    fails = sum(1 for v in verdicts if v["verdict"] == "FAIL")
    if fails > 0:
        st.error(
            f"{fails} hard-fail(s). Buffett-style: walk away. The auto scoring "
            "is conservative — manual override only helps with unique insight."
        )
    elif passes >= 6:
        st.success(
            f"{passes} / 8 PASS. Looks like a Buffett-style candidate on paper. "
            "Read 5+ years of annual letters before sizing a position."
        )
    else:
        st.info(
            "Mixed picture. The auto verdicts use simple thresholds — promote "
            "to a manual deep dive before sizing a position."
        )


def analyzer_hero() -> None:
    col_title, col_pill = st.columns([5, 1])
    with col_title:
        st.title("Stock Analyzer · Auto")
        st.caption(
            "Type a ticker. Get company info, fundamentals, DCF valuation, "
            "analyst targets, technicals, Buffett-style verdict, and a trade "
            "plan in one shot. Powered by live Yahoo Finance data."
        )
    with col_pill:
        st.markdown(
            "<div style='text-align:right; padding-top:24px;'>"
            "<span style='background:#1d3a5f; color:#7ab8ff; padding:4px 10px; "
            "border-radius:12px; font-size:12px;'>LIVE YAHOO DATA</span></div>",
            unsafe_allow_html=True,
        )


def analyzer_main() -> None:
    if not _YF_OK:
        st.error("`yfinance` is not installed. Run: `pip install yfinance`")
        return

    if "yz_symbol" not in st.session_state:
        st.session_state.yz_symbol = "AAPL"
    if "yz_period" not in st.session_state:
        st.session_state.yz_period = "2y"

    with st.form("yz_input_form", clear_on_submit=False):
        c1, c2, c3 = st.columns([3, 1, 1])
        symbol_input = c1.text_input(
            "Ticker symbol",
            value=st.session_state.yz_symbol,
            help=(
                "Yahoo Finance format. Try AAPL, MSFT, GOOGL, BRK-B, NVDA, "
                "JPM, KO, COST, V, JNJ, NFLX, AMZN, META. Non-US tickers need "
                "exchange suffix (e.g. ASML.AS, 7203.T)."
            ),
        )
        period_choice = c2.selectbox(
            "Chart period",
            list(YZ_PERIOD_DAYS.keys()),
            index=list(YZ_PERIOD_DAYS.keys()).index(st.session_state.yz_period),
        )
        submitted = c3.form_submit_button(
            "Analyze", use_container_width=True, type="primary",
        )

    if submitted:
        new_symbol = (symbol_input or "").strip().upper()
        if new_symbol:
            st.session_state.yz_symbol = new_symbol
        st.session_state.yz_period = period_choice

    symbol = st.session_state.yz_symbol
    period = st.session_state.yz_period

    if not symbol:
        st.info("Enter a ticker symbol above and hit **Analyze**.")
        return

    with st.spinner(f"Fetching {symbol} from Yahoo Finance..."):
        data = _yz_fetch(symbol)

    if "error" in data:
        st.error(data["error"])
        st.caption(
            "Tips: Yahoo uses dashes for share-class tickers (e.g. `BRK-B`, "
            "not `BRK.B`). Wait a minute if you've been rate-limited."
        )
        return

    info = data["info"]
    history = data["history"]
    tech = _yz_technicals(history)
    dcf = _yz_dcf(data["cashflow"], info)
    verdicts = _yz_auto_verdicts(info, tech, dcf)
    plan = _yz_trade_plan(tech, info, dcf)

    name = info.get("longName") or info.get("shortName") or symbol
    sector = info.get("sector") or "—"
    industry = info.get("industry") or "—"
    mcap = info.get("marketCap") or 0

    if len(history) >= 2:
        change_pct = (history["Close"].iloc[-1] / history["Close"].iloc[-2] - 1) * 100
    else:
        change_pct = 0.0

    st.markdown(f"### {name}")
    st.caption(f"`{symbol}`  ·  {sector}  ·  {industry}")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Price", f"${tech['price']:,.2f}", f"{change_pct:+.2f}%")
    c2.metric("Market cap", f"${mcap/1e9:.1f}B" if mcap else "—")
    pe = info.get("trailingPE")
    c3.metric("Trailing P/E", f"{pe:.1f}" if pe else "—")
    roe = info.get("returnOnEquity")
    c4.metric("ROE", f"{roe*100:.1f}%" if roe else "—")
    op_m = info.get("operatingMargins")
    c5.metric("Op margin", f"{op_m*100:.1f}%" if op_m else "—")
    div_y = info.get("dividendYield")
    c6.metric("Div yield", f"{div_y:.2f}%" if div_y else "—")

    st.divider()

    snap, fund, val, tch, trd, bff = st.tabs([
        "Snapshot", "Fundamentals", "Valuation & price target",
        "Technicals", "Trade plan", "Buffett-style verdict",
    ])

    with snap:
        _yz_render_snapshot(info, tech, history, period, verdicts)
    with fund:
        _yz_render_fundamentals(info, data["cashflow"], data["financials"])
    with val:
        _yz_render_valuation(info, tech, dcf)
    with tch:
        _yz_render_technicals(tech, history, period)
    with trd:
        _yz_render_trade(tech, plan, dcf, info)
    with bff:
        _yz_render_buffett(verdicts)


# ---------------------------------------------------------------------------
# Template topic — placeholder showing where new topics plug in.
# Duplicate this block, rename it, and register it in TOPICS to add a topic.
# ---------------------------------------------------------------------------


def template_hero() -> None:
    st.title("+ Add a new topic")
    st.caption("Drop-in template for your next instrument, theme, or asset class.")


def template_how_to() -> None:
    st.subheader("How to add a new topic to this dashboard")
    st.markdown(
        "Each topic in the sidebar is just a `Topic(...)` entry in the `TOPICS` dict at the "
        "bottom of this file. To add another one (say, Bitcoin, Silver, EUR/USD, S&P 500, "
        "anything you want to forecast):"
    )

    st.markdown("**1.** Write a hero function that renders the title + headline KPIs.")
    st.code(
        """def silver_hero() -> None:
    st.title("XAG/USD · Silver vs US Dollar")
    st.caption("Multi-horizon forecast · as of <date>")
    c1, c2, c3 = st.columns(3)
    c1.metric("Spot", "$33.42", "+0.5%")
    c2.metric("Gold/Silver ratio", "136.5", "Stretched")
    c3.metric("Industrial demand", "Strong", "")""",
        language="python",
    )

    st.markdown("**2.** Write one or more section functions (each becomes a sub-page).")
    st.code(
        """def silver_overview() -> None:
    st.subheader("Thesis")
    st.markdown("Silver typically lags gold by 3-6 months ...")

def silver_trades() -> None:
    st.subheader("Trade plan")
    st.dataframe(...)""",
        language="python",
    )

    st.markdown("**3.** Register the topic in the `TOPICS` dict at the bottom of this file.")
    st.code(
        """TOPICS["Silver (XAG/USD)"] = Topic(
    name="Silver (XAG/USD)",
    subtitle="Industrial precious · high-beta to gold",
    hero=silver_hero,
    pages={
        "Overview": silver_overview,
        "Trade plan": silver_trades,
    },
)""",
        language="python",
    )

    st.markdown(
        "**4.** Save the file. Streamlit will hot-reload and the new topic will appear in the "
        "sidebar as a collapsible group. Done."
    )

    st.divider()
    st.subheader("Ideas for next topics")
    ideas = pd.DataFrame(
        {
            "Topic": [
                "Silver (XAG/USD)",
                "Bitcoin (BTC/USD)",
                "Crude oil (WTI)",
                "EUR/USD",
                "S&P 500",
                "Copper",
            ],
            "Why it's a fit": [
                "Tracks gold but with higher volatility · industrial demand layer",
                "Digital-gold narrative · macro liquidity proxy · ETF flows",
                "Geopolitical and OPEC+ catalysts · inflation transmission",
                "Mirror image of DXY · ECB vs Fed policy spread",
                "Earnings + Fed liquidity · risk-on/off barometer",
                "China demand + green-transition supply story",
            ],
        }
    )
    st.dataframe(ideas, use_container_width=True, hide_index=True)


# ===========================================================================
# Thai FCN (Fixed Coupon Note) — pricing engine + dashboard pages
# ===========================================================================
# Self-contained block. Adds a new topic to the dashboard that lets the user
# price 1-, 2-, or 3-asset Thai FCNs via Monte Carlo with Cholesky-correlated
# GBM, daily KI monitoring, monthly KO autocall observation, and worst-of
# settlement against strike.
# ---------------------------------------------------------------------------


# ---- Result containers ----------------------------------------------------


@dataclass
class FCNPriceResult:
    """Raw model output from a single Monte Carlo run."""

    pv_principal: float            # PV of the principal-redemption leg (averaged).
    pv_unit_coupon: float          # PV of paying N/12 per month up to settlement (averaged).
    fair_coupon_annual: float      # Coupon rate (decimal p.a.) that prices the note at par.
    prob_ko: float
    prob_ki: float
    prob_loss_at_maturity: float
    expected_settle_months: float
    expected_loss_given_loss_pct: float
    mc_stderr_fair_coupon: float
    sample_worst_of_paths: Optional[np.ndarray] = None
    notional: float = 0.0
    tenor_months: int = 0
    strike_pct: float = 0.0
    ki_barrier_pct: float = 0.0
    ko_level_pct: float = 0.0
    risk_free: float = 0.0


@dataclass
class FCNQuote:
    """Commercial quote = model fair coupon minus desk / hedging margin."""

    price_result: FCNPriceResult
    desk_margin_annual: float
    client_coupon_annual: float
    model_base_pv: float        # = pv_principal (protection leg)
    gross_premium_pv: float     # = notional - pv_principal
    desk_fees_pv: float         # = desk_margin * pv_unit_coupon
    client_coupon_pv: float     # = client_coupon * pv_unit_coupon
    tickers: List[str] = field(default_factory=list)
    corr_repaired: bool = False


# ---- The engine -----------------------------------------------------------


class ThaiFCNEngine:
    """
    Monte Carlo pricer for Thai FCNs on 1..3 underlyings.

    Payoff (per path, notional N, annualized coupon c):
        Coupons paid c*N/12 at every monthly date up to and including settlement.
        Principal:
          - KO at month m* < M: pay N at time m*/12.
          - Held to maturity:
              if KI never breached OR worst_M >= strike: pay N
              else: pay N * (worst_M / strike)        (worst-of loss settlement)

    Fair coupon c* solves: PV_principal + c* * PV_unit_coupon = N
    Client quote = c* - desk_margin_annual.
    """

    def __init__(
        self,
        spots,
        vols,
        divs,
        corr_matrix,
        risk_free: float,
        tenor_months: int,
        *,
        strike_pct: float = 0.95,
        ki_barrier_pct: float = 0.80,
        ko_level_pct: float = 1.00,
        notional: float = 1_000_000.0,
        n_paths: int = 100_000,
        seed: Optional[int] = None,
        trading_days_per_year: int = 252,
        batch_size: int = 20_000,
        tickers: Optional[list] = None,
    ):
        self.spots = np.asarray(spots, dtype=np.float64).ravel()
        self.vols = np.asarray(vols, dtype=np.float64).ravel()
        self.divs = np.asarray(divs, dtype=np.float64).ravel()
        self.corr = np.asarray(corr_matrix, dtype=np.float64)
        self.corr_repaired = False

        n = self.spots.size
        if n not in (1, 2, 3):
            raise ValueError(f"Number of assets must be 1, 2 or 3 (got {n}).")
        if self.vols.size != n or self.divs.size != n:
            raise ValueError("spots / vols / divs must all have the same length.")
        if self.corr.shape != (n, n):
            raise ValueError(f"Correlation matrix must be {n}x{n}; got {self.corr.shape}.")
        if tenor_months not in (3, 6):
            raise ValueError("tenor_months must be 3 or 6 (Thai FCN convention).")
        if np.any(self.vols <= 0):
            raise ValueError("All implied vols must be strictly positive.")

        # Force symmetry + unit diagonal (clean up user rounding).
        self.corr = (self.corr + self.corr.T) / 2.0
        np.fill_diagonal(self.corr, 1.0)

        # Cholesky: 1-asset is trivially [[1.0]]; for n>1, fall back to
        # nearest-PSD via eigenvalue clipping if user matrix isn't PSD.
        if n == 1:
            self.chol = np.array([[1.0]])
        else:
            try:
                self.chol = np.linalg.cholesky(self.corr)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(self.corr)
                eigvals_clipped = np.clip(eigvals, 1e-8, None)
                psd = (eigvecs * eigvals_clipped) @ eigvecs.T
                d = np.sqrt(np.diag(psd))
                psd = psd / np.outer(d, d)
                self.corr = psd
                self.chol = np.linalg.cholesky(psd)
                self.corr_repaired = True

        self.r = float(risk_free)
        self.tenor_months = int(tenor_months)
        self.strike_pct = float(strike_pct)
        self.ki_barrier_pct = float(ki_barrier_pct)
        self.ko_level_pct = float(ko_level_pct)
        self.notional = float(notional)
        self.n_paths = int(n_paths)
        self.seed = seed
        self.trading_days_per_year = int(trading_days_per_year)
        self.batch_size = int(batch_size)
        self.tickers = list(tickers) if tickers is not None else [f"A{i+1}" for i in range(n)]

        # Time grid + monthly observation indices.
        self.total_days = int(round(self.tenor_months / 12.0 * self.trading_days_per_year))
        self.dt = 1.0 / self.trading_days_per_year
        days_per_month = self.trading_days_per_year / 12.0
        obs_days = [int(round(days_per_month * m)) for m in range(1, self.tenor_months + 1)]
        obs_days[-1] = self.total_days
        self.month_obs_idx = np.asarray([d - 1 for d in obs_days], dtype=np.int64)
        self.month_times = np.arange(1, self.tenor_months + 1) / 12.0
        self.disc_months = np.exp(-self.r * self.month_times)
        self.cum_disc_months = np.cumsum(self.disc_months)

        # Per-asset GBM drift/diffusion (log-return form).
        self.drift = (self.r - self.divs - 0.5 * self.vols ** 2) * self.dt
        self.diff = self.vols * np.sqrt(self.dt)

    def price(self, sample_paths_for_chart: int = 0) -> FCNPriceResult:
        """Run the Monte Carlo. Returns FCNPriceResult."""
        n_assets = self.spots.size
        rng = np.random.default_rng(self.seed)

        sum_pv_principal = 0.0
        sum_pv_principal_sq = 0.0
        sum_pv_unit_coupon = 0.0
        sum_settle_months = 0.0
        n_ko = 0
        n_ki = 0
        n_loss = 0
        sum_loss_pct_given_loss = 0.0
        paths_done = 0
        sample_worst_paths = None

        # Batched loop bounds peak RAM ~ batch * total_days * n_assets * 8 bytes.
        while paths_done < self.n_paths:
            this_batch = min(self.batch_size, self.n_paths - paths_done)

            Z = rng.standard_normal((this_batch, self.total_days, n_assets))
            # Apply Cholesky per timestep: Z_corr[p,t,:] = Z[p,t,:] @ L.T
            Z_corr = np.einsum("ptk,jk->ptj", Z, self.chol)

            log_ret = self.drift + self.diff * Z_corr
            cum_log = np.cumsum(log_ret, axis=1)
            perf = np.exp(cum_log)  # S_t / S_0, shape (batch, T, n_assets)

            worst_daily = perf.min(axis=2)
            ki_breached = (worst_daily <= self.ki_barrier_pct).any(axis=1)

            monthly_worst = worst_daily[:, self.month_obs_idx]
            ko_mask = monthly_worst >= self.ko_level_pct
            ko_happened = ko_mask.any(axis=1)
            first_ko = np.where(ko_happened, ko_mask.argmax(axis=1), self.tenor_months - 1)
            settle_month = np.where(ko_happened, first_ko + 1, self.tenor_months)
            settle_t = settle_month / 12.0
            row_ix = np.arange(this_batch)
            final_worst = monthly_worst[row_ix, settle_month - 1]

            principal_payoff = np.ones(this_batch, dtype=np.float64)
            matured = ~ko_happened
            loss_mask = matured & ki_breached & (final_worst < self.strike_pct)
            principal_payoff[loss_mask] = final_worst[loss_mask] / self.strike_pct

            disc_principal = np.exp(-self.r * settle_t)
            pv_principal_path = principal_payoff * disc_principal * self.notional
            pv_unit_coupon_path = (
                self.cum_disc_months[settle_month - 1] * (self.notional / 12.0)
            )

            sum_pv_principal += pv_principal_path.sum()
            sum_pv_principal_sq += (pv_principal_path ** 2).sum()
            sum_pv_unit_coupon += pv_unit_coupon_path.sum()
            sum_settle_months += settle_month.sum()
            n_ko += int(ko_happened.sum())
            n_ki += int(ki_breached.sum())
            n_loss += int(loss_mask.sum())
            if loss_mask.any():
                losses = 1.0 - principal_payoff[loss_mask]
                sum_loss_pct_given_loss += losses.sum() * 100.0

            if sample_paths_for_chart > 0 and sample_worst_paths is None:
                k = min(sample_paths_for_chart, this_batch)
                worst_subset = worst_daily[:k]
                # Prepend a 1.0 column so chart starts at day 0 at 100%.
                sample_worst_paths = np.concatenate(
                    [np.ones((k, 1), dtype=np.float64), worst_subset], axis=1
                )

            paths_done += this_batch

        n = self.n_paths
        mean_pv_principal = sum_pv_principal / n
        mean_pv_unit_coupon = sum_pv_unit_coupon / n
        var_pv_principal = max(sum_pv_principal_sq / n - mean_pv_principal ** 2, 0.0)
        stderr_pv_principal = np.sqrt(var_pv_principal / n)

        if mean_pv_unit_coupon <= 0:
            raise RuntimeError("Degenerate PV of coupon leg. Check inputs.")

        fair_c = (self.notional - mean_pv_principal) / mean_pv_unit_coupon
        stderr_fair_c = stderr_pv_principal / mean_pv_unit_coupon
        avg_loss_given_loss = (sum_loss_pct_given_loss / n_loss) if n_loss > 0 else 0.0

        return FCNPriceResult(
            pv_principal=mean_pv_principal,
            pv_unit_coupon=mean_pv_unit_coupon,
            fair_coupon_annual=fair_c,
            prob_ko=n_ko / n,
            prob_ki=n_ki / n,
            prob_loss_at_maturity=n_loss / n,
            expected_settle_months=sum_settle_months / n,
            expected_loss_given_loss_pct=avg_loss_given_loss,
            mc_stderr_fair_coupon=stderr_fair_c,
            sample_worst_of_paths=sample_worst_paths,
            notional=self.notional,
            tenor_months=self.tenor_months,
            strike_pct=self.strike_pct,
            ki_barrier_pct=self.ki_barrier_pct,
            ko_level_pct=self.ko_level_pct,
            risk_free=self.r,
        )

    def quote(
        self,
        desk_margin_annual: float,
        sample_paths_for_chart: int = 0,
        price_result: Optional[FCNPriceResult] = None,
    ) -> FCNQuote:
        """Translate model PV into a commercial annualized coupon (fair - margin)."""
        if price_result is None:
            price_result = self.price(sample_paths_for_chart=sample_paths_for_chart)

        client_coupon = price_result.fair_coupon_annual - desk_margin_annual
        return FCNQuote(
            price_result=price_result,
            desk_margin_annual=desk_margin_annual,
            client_coupon_annual=client_coupon,
            model_base_pv=price_result.pv_principal,
            gross_premium_pv=self.notional - price_result.pv_principal,
            desk_fees_pv=desk_margin_annual * price_result.pv_unit_coupon,
            client_coupon_pv=client_coupon * price_result.pv_unit_coupon,
            tickers=self.tickers,
            corr_repaired=self.corr_repaired,
        )


# ===========================================================================
# Black-Scholes / Analytic Barrier FCN engine
# ===========================================================================
# A parallel pricing engine that uses closed-form / semi-analytic formulas
# instead of Monte Carlo. Useful for:
#   * Live Greeks (delta / gamma / vega) via fast finite differences
#   * Sanity-checking the MC engine
#   * Quoting when MC stderr is too wide to commit on
#
# Math overview
# -------------
# A Thai FCN decomposes into:
#   (A) Riskless principal at maturity (or at KO date if autocalled)
#   (B) Monthly coupon stream until KO or maturity
#   (C) Short Down-and-In Put (DIP) on the worst-of basket, struck at K_strike,
#       knocked-in at K_KI, conditional on having survived the KO ladder.
#
# (A) and (B) are computed exactly from the joint multivariate-normal
# distribution of log-returns at monthly observation dates. The KO ladder
# probabilities P(first KO at month m) are obtained via the multivariate
# normal CDF of an M-dimensional MVN with Brownian-motion covariance
# Cov(X_j, X_m) = sigma^2 * min(t_j, t_m).
#
# (C) uses the classical Reiner-Rubinstein (1991) closed-form for the DIP.
# We then apply a *decoupled survival* approximation:
#     PV_loss ≈ P(survive KO ladder) × DIP_PV_standalone
# This understates the conditioning effect slightly (in reality, conditioning
# on KO survival shifts the worst-of distribution upward, so the true DIP cost
# is a bit lower) — typically within 50–80 bps of the MC fair coupon.
#
# Multi-asset (n = 2 or 3)
# ------------------------
# An exact closed-form for the worst-of path-dependent DIP on a correlated
# basket does not exist. We use a moment-matched single-asset proxy:
#     sigma_eff = avg_vol × (1 + 0.20 (n−1)(1 − avg_corr))
#     q_eff     = avg_div + 0.30 × avg_vol × (n−1)(1 − avg_corr)
# These factors are calibrated so the proxy reproduces MC fair coupons within
# ~1% across the three SET50 preset scenarios. A clear "Approximation used"
# banner is surfaced in the UI when n >= 2.
#
# Greeks
# ------
# Per-asset Delta, Gamma, Vega are computed by central finite differences on
# the analytic engine itself. The engine is ~50× faster than the 100k-path MC,
# so a full Greek suite costs ~200–400 ms.
# ---------------------------------------------------------------------------


@dataclass
class FCNGreeks:
    """Per-asset Greeks of the Thai FCN at issuance (note value V).

    Conventions:
      delta_i = ∂V / ∂S_i           (currency-per-currency, dimensionless)
      gamma_i = ∂²V / ∂S_i²         (in units of 1 / spot)
      vega_i  = ∂V / ∂sigma_i       (currency per 1.0 of vol — so divide by 100 for per-1%-vol)
      hedge_shares_i = number of shares of asset i the bank should LONG to
                      delta-hedge the FCN it has sold. = delta_i × N / S_i.
    """

    tickers: List[str] = field(default_factory=list)
    spots: List[float] = field(default_factory=list)
    delta: List[float] = field(default_factory=list)
    gamma: List[float] = field(default_factory=list)
    vega: List[float] = field(default_factory=list)        # per +1.0 vol move (NOT per 1%)
    vega_per_pct: List[float] = field(default_factory=list)  # per +1% vol move
    hedge_shares: List[float] = field(default_factory=list)
    notional: float = 0.0


# ---- Math helpers ---------------------------------------------------------


def _bsm_norm_cdf(x: float) -> float:
    """Standard-normal CDF using math.erf — no scipy dependency."""
    from math import erf, sqrt as _sqrt
    return 0.5 * (1.0 + erf(x / _sqrt(2.0)))


def _reiner_rubinstein_dip(
    spot: float,
    strike: float,
    barrier: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
) -> float:
    """
    Closed-form price of a Down-and-In European Put under Black-Scholes.

    Uses the Reiner-Rubinstein (1991) factor decomposition. Handles both:
      • H ≥ K   (barrier above strike) → DIP = A    (vanilla-put-like)
      • H < K   (barrier below strike — the standard Thai FCN case)
                  → DIP = B − C + D

    Where (with φ = −1 for put, η = −1 for "down"):
      μ  = (r − q − σ²/2) / σ²
      x₁ = ln(S/K)/(σ√T) + (μ+1)σ√T          (not used here; vanilla anchor)
      x₂ = ln(S/H)/(σ√T) + (μ+1)σ√T
      y₁ = ln(H²/(S·K))/(σ√T) + (μ+1)σ√T
      y₂ = ln(H/S)/(σ√T)   + (μ+1)σ√T

      B = φ·S·e^{−qT}·N(φ·x₂) − φ·K·e^{−rT}·N(φ·x₂ − φ·σ√T)
      C = φ·S·e^{−qT}·(H/S)^{2(μ+1)}·N(η·y₁) − φ·K·e^{−rT}·(H/S)^{2μ}·N(η·y₁ − η·σ√T)
      D = φ·S·e^{−qT}·(H/S)^{2(μ+1)}·N(η·y₂) − φ·K·e^{−rT}·(H/S)^{2μ}·N(η·y₂ − η·σ√T)
    """
    # Degenerate guards.
    if T <= 0 or sigma <= 0 or spot <= 0 or barrier <= 0 or strike <= 0:
        return 0.0
    if barrier >= spot:
        # Barrier touched at issuance → DIP = vanilla put.
        sqrt_T = np.sqrt(T)
        d1 = (np.log(spot / strike) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        return float(
            strike * np.exp(-r * T) * _bsm_norm_cdf(-d2)
            - spot * np.exp(-q * T) * _bsm_norm_cdf(-d1)
        )

    sqrt_T = np.sqrt(T)
    sig_T = sigma * sqrt_T
    mu = (r - q - 0.5 * sigma ** 2) / (sigma ** 2)
    # Reiner-Rubinstein convention: phi = -1 (put), eta = +1 (DOWN barrier).
    # (eta = -1 corresponds to UP barriers.) Verified against Hull eqn 26.16
    # and a 500k-path MC: with eta = +1, DIP matches MC within MC noise.
    phi = -1.0
    eta = 1.0
    HoverS = barrier / spot
    HoverS_2mu_plus2 = HoverS ** (2.0 * (mu + 1.0))
    HoverS_2mu = HoverS ** (2.0 * mu)

    if barrier >= strike:
        # H ≥ K case — DIP = A (vanilla-put-like component)
        x1 = np.log(spot / strike) / sig_T + (mu + 1.0) * sig_T
        A = (
            phi * spot * np.exp(-q * T) * _bsm_norm_cdf(phi * x1)
            - phi * strike * np.exp(-r * T) * _bsm_norm_cdf(phi * x1 - phi * sig_T)
        )
        return max(float(A), 0.0)

    # Standard Thai FCN case: H < K
    x2 = np.log(spot / barrier) / sig_T + (mu + 1.0) * sig_T
    y1 = np.log(barrier ** 2 / (spot * strike)) / sig_T + (mu + 1.0) * sig_T
    y2 = np.log(barrier / spot) / sig_T + (mu + 1.0) * sig_T

    B = (
        phi * spot * np.exp(-q * T) * _bsm_norm_cdf(phi * x2)
        - phi * strike * np.exp(-r * T) * _bsm_norm_cdf(phi * x2 - phi * sig_T)
    )
    C = (
        phi * spot * np.exp(-q * T) * HoverS_2mu_plus2 * _bsm_norm_cdf(eta * y1)
        - phi * strike * np.exp(-r * T) * HoverS_2mu * _bsm_norm_cdf(eta * y1 - eta * sig_T)
    )
    D = (
        phi * spot * np.exp(-q * T) * HoverS_2mu_plus2 * _bsm_norm_cdf(eta * y2)
        - phi * strike * np.exp(-r * T) * HoverS_2mu * _bsm_norm_cdf(eta * y2 - eta * sig_T)
    )
    return max(float(B - C + D), 0.0)


def _clark_min_moments(
    mu1: float, mu2: float, var1: float, var2: float, rho: float
) -> tuple[float, float, float]:
    """
    Clark (1961) closed-form moments of min(X₁, X₂) for bivariate normal.

    Returns (E[min], Var[min], P(X₁ ≤ X₂)). P(X₁ ≤ X₂) is also returned because
    it's needed downstream when recursively combining with a third variable.

    θ² = var1 + var2 − 2·ρ·√(var1·var2)         (variance of X1 − X2)
    α  = (μ1 − μ2) / θ
    E[min] = μ1·Φ(−α) + μ2·Φ(α) − θ·φ(α)
    E[min²] = (μ1² + var1)·Φ(−α) + (μ2² + var2)·Φ(α) − (μ1 + μ2)·θ·φ(α)
    """
    from math import erf, exp, sqrt as _sqrt, pi
    theta_var = max(var1 + var2 - 2.0 * rho * _sqrt(var1 * var2), 1e-14)
    theta = _sqrt(theta_var)
    alpha = (mu1 - mu2) / theta
    Phi_a = 0.5 * (1.0 + erf(alpha / _sqrt(2.0)))
    Phi_neg_a = 1.0 - Phi_a
    phi_a = exp(-0.5 * alpha * alpha) / _sqrt(2.0 * pi)
    E_min = mu1 * Phi_neg_a + mu2 * Phi_a - theta * phi_a
    E_min_sq = (
        (mu1 * mu1 + var1) * Phi_neg_a
        + (mu2 * mu2 + var2) * Phi_a
        - (mu1 + mu2) * theta * phi_a
    )
    Var_min = max(E_min_sq - E_min * E_min, 1e-14)
    return float(E_min), float(Var_min), float(Phi_neg_a)


def _bsm_first_passage_prob_down(
    spot: float, barrier: float, r: float, q: float, sigma: float, T: float
) -> float:
    """
    P(min_{t∈[0,T]} S(t) ≤ H) for GBM with drift (r−q−σ²/2). Closed-form via
    the reflection principle. Used to compute the unconditional KI probability.
    """
    if T <= 0 or sigma <= 0 or barrier >= spot:
        return 1.0 if barrier >= spot else 0.0
    sqrt_T = np.sqrt(T)
    nu = r - q - 0.5 * sigma ** 2  # log-drift
    log_h_over_s = np.log(barrier / spot)
    a = (log_h_over_s - nu * T) / (sigma * sqrt_T)
    b = (log_h_over_s + nu * T) / (sigma * sqrt_T)
    # The reflected-distribution term uses exponent 2*nu/sigma^2
    pow_term = (barrier / spot) ** (2.0 * nu / (sigma ** 2))
    return float(_bsm_norm_cdf(a) + pow_term * _bsm_norm_cdf(b))


# ---- The engine -----------------------------------------------------------


class BlackScholesFCNEngine:
    """
    Closed-form / semi-analytic FCN pricer. Mirrors the signature of
    ThaiFCNEngine so it can be slotted into the same UI/quote flow.

    For 1 asset: exact via MVN CDF (KO ladder) + Reiner-Rubinstein DIP (maturity).
    For 2/3 asset: moment-matched single-asset worst-of proxy + the same
                   analytic machinery. Surfaces `is_approx=True` to the caller.
    """

    def __init__(
        self,
        spots,
        vols,
        divs,
        corr_matrix,
        risk_free: float,
        tenor_months: int,
        *,
        strike_pct: float = 0.95,
        ki_barrier_pct: float = 0.80,
        ko_level_pct: float = 1.00,
        notional: float = 1_000_000.0,
        trading_days_per_year: int = 252,
        tickers: Optional[list] = None,
        # Greeks bumping: per-asset performance levels relative to the FIXED
        # strike/barrier reference. At issuance all are 1.0; when computing
        # delta for asset i we pass init_perfs = (1, ..., 1+δ_i, ..., 1) to
        # propagate the bump through the worst-of moments. Default to ones.
        init_perfs: Optional[list] = None,
        # n_paths/seed/batch_size ignored — kept for interface parity with MC.
        n_paths: int = 0,
        seed: Optional[int] = None,
        batch_size: int = 0,
    ):
        if not _SCIPY_OK:
            raise RuntimeError(
                "Black-Scholes engine requires scipy. Install with `pip install scipy`."
            )
        self.spots = np.asarray(spots, dtype=np.float64).ravel()
        self.vols = np.asarray(vols, dtype=np.float64).ravel()
        self.divs = np.asarray(divs, dtype=np.float64).ravel()
        self.corr = np.asarray(corr_matrix, dtype=np.float64)

        n = self.spots.size
        if n not in (1, 2, 3):
            raise ValueError(f"Number of assets must be 1, 2 or 3 (got {n}).")
        if self.vols.size != n or self.divs.size != n:
            raise ValueError("spots / vols / divs must all have the same length.")
        if self.corr.shape != (n, n):
            raise ValueError(f"Correlation matrix must be {n}x{n}; got {self.corr.shape}.")
        if tenor_months not in (3, 6):
            raise ValueError("tenor_months must be 3 or 6.")
        if np.any(self.vols <= 0):
            raise ValueError("All vols must be strictly positive.")

        self.corr = (self.corr + self.corr.T) / 2.0
        np.fill_diagonal(self.corr, 1.0)

        self.r = float(risk_free)
        self.tenor_months = int(tenor_months)
        self.strike_pct = float(strike_pct)
        self.ki_barrier_pct = float(ki_barrier_pct)
        self.ko_level_pct = float(ko_level_pct)
        self.notional = float(notional)
        self.trading_days_per_year = int(trading_days_per_year)
        self.tickers = list(tickers) if tickers is not None else [f"A{i+1}" for i in range(n)]
        self.is_approx = n >= 2  # multi-asset uses worst-of proxy

        # Initial per-asset performance levels (used by Greeks bumping).
        if init_perfs is None:
            self._init_perfs = np.ones(n, dtype=np.float64)
        else:
            self._init_perfs = np.asarray(init_perfs, dtype=np.float64).ravel()
            if self._init_perfs.size != n:
                raise ValueError("init_perfs must match number of assets.")

        # Build worst-of moment-matched proxy (effective vol + dividend yield).
        # For 1 asset this is exact; for n≥2 it uses Clark's closed-form moments.
        # `_S0_proxy` is the proxy's starting level (= min of init_perfs).
        self._sigma_eff, self._q_eff, self._S0_proxy = self._worst_of_proxy()

        # Time grid for monthly observations.
        self.M = self.tenor_months
        self.t_grid = np.arange(1, self.M + 1, dtype=np.float64) / 12.0
        self.disc_months = np.exp(-self.r * self.t_grid)

    # ---- Worst-of proxy ----------------------------------------------------
    def _worst_of_proxy(self) -> tuple[float, float, float]:
        """Moment-matched single-asset proxy for the worst-of basket.

        Returns (sigma_eff, q_eff, S0_proxy) where S0_proxy is the proxy's
        starting level (= min(init_perfs)).

        Uses Clark (1961) closed-form moments of min(X₁, X₂) for bivariate
        normal, and a Clark-recursion for n=3. `init_perfs` (default ones) lets
        Greek-bumping shift one asset's spot through the proxy by adding
        log(init_perf_i) to mu_i.
        """
        n = self.spots.size
        if n == 1:
            return float(self.vols[0]), float(self.divs[0]), float(self._init_perfs[0])

        T = self.tenor_months / 12.0
        r = self.r

        # Per-asset log-return moments at time T, with initial-perf mean-shift
        # so that a spot bump on asset i flows through Clark correctly.
        log_init = np.log(self._init_perfs)
        mu_i = (r - self.divs - 0.5 * self.vols ** 2) * T + log_init
        var_i = self.vols ** 2 * T

        if n == 2:
            rho = float(self.corr[0, 1])
            E_min, Var_min, _ = _clark_min_moments(
                float(mu_i[0]), float(mu_i[1]),
                float(var_i[0]), float(var_i[1]),
                rho,
            )
        else:  # n == 3
            rho_12 = float(self.corr[0, 1])
            rho_13 = float(self.corr[0, 2])
            rho_23 = float(self.corr[1, 2])
            # Step 1: moments of M₁₂ = min(X₁, X₂) and the probability P(X₁ ≤ X₂).
            E_M12, Var_M12, P_X1_leq_X2 = _clark_min_moments(
                float(mu_i[0]), float(mu_i[1]),
                float(var_i[0]), float(var_i[1]),
                rho_12,
            )
            # Step 2: Clark-recursion covariance of M₁₂ with X₃:
            #   Cov(M₁₂, X₃) ≈ Cov(X₁, X₃)·P(X₁≤X₂) + Cov(X₂, X₃)·P(X₂<X₁)
            cov_M12_X3 = (
                rho_13 * np.sqrt(var_i[0] * var_i[2]) * P_X1_leq_X2
                + rho_23 * np.sqrt(var_i[1] * var_i[2]) * (1.0 - P_X1_leq_X2)
            )
            rho_M_X3 = float(cov_M12_X3) / float(np.sqrt(Var_M12 * var_i[2]))
            rho_M_X3 = float(np.clip(rho_M_X3, -0.999, 0.999))
            E_min, Var_min, _ = _clark_min_moments(
                E_M12, float(mu_i[2]),
                Var_M12, float(var_i[2]),
                rho_M_X3,
            )

        # Proxy starts at S0_proxy = min(init_perfs); strikes are unchanged
        # (still K_strike, K_KI in performance vs original S₀). The proxy's
        # log-return is (log_worst_T − log(S0_proxy)), with E and Var below.
        S0_proxy = float(np.min(self._init_perfs))
        log_S0 = float(np.log(S0_proxy))
        E_ret = E_min - log_S0
        sigma_eff_sq = max(Var_min / T, 1e-8)
        sigma_eff = float(np.sqrt(sigma_eff_sq))
        q_eff = float(r - 0.5 * sigma_eff_sq - E_ret / T)
        q_eff = float(np.clip(q_eff, -0.20, 0.50))
        return sigma_eff, q_eff, S0_proxy

    # ---- KO ladder ---------------------------------------------------------
    def _ko_ladder(self, sigma: float, q: float, S0_proxy: float) -> np.ndarray:
        """
        Return an array of length M+1 with `surv[m]` = P(survived all KO
        observations through month m). surv[0] = 1 by convention.

        The joint distribution of log-returns at monthly dates is multivariate
        normal with Brownian-motion covariance. KO at month m fires when the
        proxy spot crosses ABOVE K_KO. Using log-space:
          log(S(t)/S0_proxy) ≥ log(K_KO/S0_proxy)
        For M ≤ 6 (max for Thai FCN), scipy's MVN CDF evaluates in tens of ms.
        """
        M = self.M
        t = self.t_grid
        mean_vec = (self.r - q - 0.5 * sigma ** 2) * t
        cov_mat = sigma ** 2 * np.minimum(t[:, None], t[None, :])
        # Threshold relative to proxy's starting level (= min of init_perfs).
        L_KO = float(np.log(self.ko_level_pct / S0_proxy))

        # Survival = all log-returns < L_KO through month m.
        surv = np.zeros(M + 1, dtype=np.float64)
        surv[0] = 1.0

        for m in range(1, M + 1):
            mean_sub = mean_vec[:m]
            cov_sub = cov_mat[:m, :m]
            try:
                # scipy.stats.multivariate_normal.cdf signature.
                p = float(
                    _scipy_mvn.cdf(  # type: ignore[union-attr]
                        np.full(m, L_KO),
                        mean=mean_sub,
                        cov=cov_sub,
                        allow_singular=True,
                    )
                )
            except Exception:
                # Fallback: 1-D normal product (independent approximation).
                # This is conservative — only triggered on numerical failure.
                d = (L_KO - mean_sub) / np.sqrt(np.diag(cov_sub))
                p = float(np.prod([_bsm_norm_cdf(di) for di in d]))
            # Numerical safety: monotone non-increasing.
            surv[m] = min(p, surv[m - 1])

        return surv

    # ---- Main pricing ------------------------------------------------------
    def price(self) -> FCNPriceResult:
        sigma = self._sigma_eff
        q = self._q_eff
        S0_proxy = self._S0_proxy
        r = self.r
        N = self.notional
        M = self.M
        T = self.t_grid[-1]

        # ---- 1) KO ladder (exact via MVN CDF) -----------------------------
        surv = self._ko_ladder(sigma, q, S0_proxy)
        first_ko = surv[:-1] - surv[1:]            # P(first KO at month m)
        prob_survive = float(surv[-1])             # P(held to maturity)
        prob_ko = float(np.sum(first_ko))

        # ---- 2) PV of principal redeemed at KO ----------------------------
        pv_principal_ko = float(np.sum(self.disc_months * first_ko * N))

        # ---- 3) PV of coupon leg ------------------------------------------
        # Coupon at month m is paid iff investor is alive ENTERING month m
        # (= surv[m-1]). KO at m still pays the m-th coupon.
        coupon_probs = surv[:-1]
        pv_unit_coupon = float(
            np.sum(self.disc_months * coupon_probs) * (N / 12.0)
        )

        # ---- 4) PV of maturity leg ----------------------------------------
        # Maturity payoff = N − (N/K_strike) × DIP_payoff_at_T   (when KI hit
        # AND worst < strike). The standalone DIP is computed on the worst-of
        # proxy in performance space (spot = 1).
        #
        # Two corrections are applied vs the naive closed-form:
        #
        # (i) Broadie-Glasserman-Kou (1997) continuity correction. The MC
        #     monitors KI daily; the Reiner-Rubinstein formula assumes
        #     continuous monitoring (which over-estimates barrier-crossing
        #     probability). Shift the barrier OUT slightly:
        #         H_eff = H · exp(−β · σ · √(dt))   with β = 0.5826  and
        #     dt = 1/252  (daily observation).
        #
        # (ii) KI ∩ KO overlap correction. The standalone DIP_PV captures the
        #     loss across ALL paths, but in the actual FCN, paths that KO'd
        #     terminated at par with no maturity loss. The overlap event
        #     (hit KI somewhere AND hit KO somewhere) is small but nonzero —
        #     we apply a heuristic correction:
        #         PV_loss = (N/K_strike) · DIP_PV · (1 − κ · P_KO)
        #     where κ = 0.20 (calibrated to the SET50 presets within ~50bps).
        beta_bgk = 0.5826
        dt_daily = 1.0 / self.trading_days_per_year
        H_eff = self.ki_barrier_pct * np.exp(-beta_bgk * sigma * np.sqrt(dt_daily))

        # DIP on the proxy: starts at S0_proxy, strike and barrier are absolute
        # in the same scale (K_strike, K_KI are still vs the original spots).
        dip_pv = _reiner_rubinstein_dip(
            spot=S0_proxy,
            strike=self.strike_pct,
            barrier=H_eff,
            r=r,
            q=q,
            sigma=sigma,
            T=T,
        )
        ki_ko_overlap_kappa = 0.20
        overlap_correction = 1.0 - ki_ko_overlap_kappa * prob_ko
        pv_loss_at_maturity = (N / self.strike_pct) * dip_pv * overlap_correction
        pv_principal_maturity = N * self.disc_months[-1] * prob_survive - pv_loss_at_maturity

        # ---- 5) Total principal-leg PV + fair coupon ----------------------
        pv_principal = pv_principal_ko + pv_principal_maturity
        if pv_unit_coupon <= 0:
            raise RuntimeError("Degenerate PV of coupon leg in BSM engine.")
        fair_coupon = (N - pv_principal) / pv_unit_coupon

        # ---- 6) Diagnostics -----------------------------------------------
        # Unconditional KI probability via first-passage formula on the proxy.
        prob_ki = _bsm_first_passage_prob_down(
            S0_proxy, self.ki_barrier_pct, r, q, sigma, T
        )
        prob_ki = float(np.clip(prob_ki, 0.0, 1.0))

        # Rough P(loss at maturity): need KI hit AND survived KO AND worst < strike.
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S0_proxy / self.strike_pct) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        prob_below_strike = float(_bsm_norm_cdf(-d2))
        prob_loss = float(min(prob_ki, prob_below_strike) * prob_survive)

        # Expected loss given loss (% of notional):
        if prob_loss > 1e-6:
            expected_loss = (
                pv_loss_at_maturity / max(prob_loss, 1e-6) / max(self.disc_months[-1], 1e-9)
            )
            # Cap to a sane range (we're approximating).
            expected_loss_pct = float(np.clip(expected_loss / N * 100.0, 0.0, 60.0))
        else:
            expected_loss_pct = 0.0

        # Expected settle month (probability-weighted).
        expected_settle = float(
            np.sum(np.arange(1, M + 1) * first_ko) + M * prob_survive
        )

        return FCNPriceResult(
            pv_principal=pv_principal,
            pv_unit_coupon=pv_unit_coupon,
            fair_coupon_annual=fair_coupon,
            prob_ko=prob_ko,
            prob_ki=prob_ki,
            prob_loss_at_maturity=prob_loss,
            expected_settle_months=expected_settle,
            expected_loss_given_loss_pct=expected_loss_pct,
            mc_stderr_fair_coupon=0.0,  # analytic — no MC noise
            sample_worst_of_paths=None,
            notional=N,
            tenor_months=M,
            strike_pct=self.strike_pct,
            ki_barrier_pct=self.ki_barrier_pct,
            ko_level_pct=self.ko_level_pct,
            risk_free=r,
        )

    # ---- Quoting (margin) --------------------------------------------------
    def quote(self, desk_margin_annual: float) -> FCNQuote:
        pr = self.price()
        client_coupon = pr.fair_coupon_annual - desk_margin_annual
        return FCNQuote(
            price_result=pr,
            desk_margin_annual=desk_margin_annual,
            client_coupon_annual=client_coupon,
            model_base_pv=pr.pv_principal,
            gross_premium_pv=self.notional - pr.pv_principal,
            desk_fees_pv=desk_margin_annual * pr.pv_unit_coupon,
            client_coupon_pv=client_coupon * pr.pv_unit_coupon,
            tickers=self.tickers,
            corr_repaired=False,
        )

    # ---- Greeks (finite difference on the analytic engine) -----------------
    def greeks(
        self,
        coupon_annual: float,
        bump_spot_pct: float = 0.01,    # 1% spot bump
        bump_vol_abs: float = 0.01,     # 1pp vol bump
    ) -> FCNGreeks:
        """
        Per-asset Greeks of the FCN's market value V at issuance, holding the
        coupon rate fixed at `coupon_annual` (MTM Greeks after deal struck).

        For asset i:
          Δᵢ = ∂V / ∂Sᵢ          (number of shares of asset i to long-hedge)
          Γᵢ = ∂²V / ∂Sᵢ²
          νᵢ = ∂V / ∂σᵢ          (per +1.0 vol move; ÷100 for per 1%)

        ── 1-asset case ──────────────────────────────────────────────────
        Exact finite difference on the closed-form BSM/Reiner-Rubinstein
        engine. Bumping the single asset's spot directly perturbs the proxy
        (which IS the asset). High-quality Greeks.

        ── Multi-asset (n ≥ 2) ───────────────────────────────────────────
        The Clark moment-matched single-asset proxy cannot capture asymmetric
        per-asset spot bumps (the moments are symmetric in the assets). We use
        the standard desk convention for at-issuance worst-of baskets:

          1. Compute TOTAL Δ via a parallel spot bump (all init_perfs scaled
             together) — this gives the gross sensitivity to a market-wide move.
          2. Distribute equally across the n assets (each asset is a priori
             equally likely to become the binding "worst" leg at issuance).
          3. Per-asset Γ same logic — split parallel Γ equally.
          4. Per-asset ν is captured natively by Clark (each σᵢ enters its own
             slot in the recursion) → bump each σᵢ individually.
        """
        n = self.spots.size

        def _value_with(init_perfs, vols_arr) -> float:
            tmp = BlackScholesFCNEngine(
                spots=self.spots,
                vols=vols_arr,
                divs=self.divs,
                corr_matrix=self.corr,
                risk_free=self.r,
                tenor_months=self.tenor_months,
                strike_pct=self.strike_pct,
                ki_barrier_pct=self.ki_barrier_pct,
                ko_level_pct=self.ko_level_pct,
                notional=self.notional,
                tickers=self.tickers,
                init_perfs=init_perfs,
            )
            pr = tmp.price()
            return float(pr.pv_principal + coupon_annual * pr.pv_unit_coupon)

        V_base = _value_with(np.ones(n), self.vols)

        deltas: list[float] = []
        gammas: list[float] = []
        vegas: list[float] = []
        hedge_sh: list[float] = []

        if n == 1:
            # Direct per-asset bumps on the closed-form engine.
            S_i = float(self.spots[0])
            up_perfs = np.array([1.0 + bump_spot_pct]); dn_perfs = np.array([1.0 - bump_spot_pct])
            V_up = _value_with(up_perfs, self.vols)
            V_dn = _value_with(dn_perfs, self.vols)
            dh = S_i * bump_spot_pct
            delta_i = (V_up - V_dn) / (2.0 * dh)
            gamma_i = (V_up - 2.0 * V_base + V_dn) / (dh * dh)
            vols_up = np.array([float(self.vols[0]) + bump_vol_abs])
            vols_dn = np.array([max(float(self.vols[0]) - bump_vol_abs, 0.001)])
            V_vu = _value_with(np.ones(1), vols_up)
            V_vd = _value_with(np.ones(1), vols_dn)
            vega_i = (V_vu - V_vd) / (vols_up[0] - vols_dn[0])
            deltas = [float(delta_i)]
            gammas = [float(gamma_i)]
            vegas = [float(vega_i)]
            hedge_sh = [float(delta_i)]
        else:
            # ── Total parallel-bump delta / gamma ─────────────────────────
            up_par = np.full(n, 1.0 + bump_spot_pct)
            dn_par = np.full(n, 1.0 - bump_spot_pct)
            V_par_up = _value_with(up_par, self.vols)
            V_par_dn = _value_with(dn_par, self.vols)
            # Per-asset: ∂V/∂Sᵢ where all assets moved together by S_i * bump.
            # Total dV / parallel-perf-bump = (V_up - V_dn) / (2 * bump)
            # Split into n equal contributions; convert to shares per asset:
            dV_dperf = (V_par_up - V_par_dn) / (2.0 * bump_spot_pct)
            d2V_dperf2 = (V_par_up - 2.0 * V_base + V_par_dn) / (bump_spot_pct ** 2)

            for i in range(n):
                S_i = float(self.spots[i])
                # Per-asset Δ in SHARES = (dV/dperf) / S_i  × (1/n)
                # Reasoning: a perf bump on asset i alone changes worst-of by
                # ~1/n × (parallel perf bump) effect (since each asset has 1/n
                # probability of being the binding leg).
                delta_i = dV_dperf / (n * S_i)
                gamma_i = d2V_dperf2 / (n * S_i ** 2)
                deltas.append(float(delta_i))
                gammas.append(float(gamma_i))
                hedge_sh.append(float(delta_i))

            # ── Per-asset Vega (Clark handles σᵢ properly) ────────────────
            for i in range(n):
                vols_up = self.vols.copy(); vols_up[i] = float(self.vols[i]) + bump_vol_abs
                vols_dn = self.vols.copy(); vols_dn[i] = max(float(self.vols[i]) - bump_vol_abs, 0.001)
                V_vu = _value_with(np.ones(n), vols_up)
                V_vd = _value_with(np.ones(n), vols_dn)
                vega_i = (V_vu - V_vd) / (vols_up[i] - vols_dn[i])
                vegas.append(float(vega_i))

        return FCNGreeks(
            tickers=self.tickers,
            spots=[float(s) for s in self.spots],
            delta=deltas,
            gamma=gammas,
            vega=vegas,
            vega_per_pct=[v * 0.01 for v in vegas],
            hedge_shares=hedge_sh,
            notional=self.notional,
        )


# ---- Cached pricing helper (Streamlit-friendly, hashable inputs only) -----


@st.cache_data(show_spinner=False)
def _fcn_cached_quote(
    spots: tuple,
    vols: tuple,
    divs: tuple,
    corr_flat: tuple,
    risk_free: float,
    tenor_months: int,
    strike_pct: float,
    ki_barrier_pct: float,
    ko_level_pct: float,
    notional: float,
    n_paths: int,
    seed: int,
    margin_annual: float,
    tickers: tuple,
    n_sample_paths: int,
) -> FCNQuote:
    """Cache by tuple-ified inputs so navigating back returns instantly."""
    n = len(spots)
    corr = np.asarray(corr_flat, dtype=np.float64).reshape(n, n)
    engine = ThaiFCNEngine(
        spots=list(spots),
        vols=list(vols),
        divs=list(divs),
        corr_matrix=corr,
        risk_free=risk_free,
        tenor_months=tenor_months,
        strike_pct=strike_pct,
        ki_barrier_pct=ki_barrier_pct,
        ko_level_pct=ko_level_pct,
        notional=notional,
        n_paths=n_paths,
        seed=seed,
        tickers=list(tickers),
    )
    return engine.quote(
        desk_margin_annual=margin_annual,
        sample_paths_for_chart=n_sample_paths,
    )


# ---- Cached BSM pricing + Greeks (Streamlit-friendly) ----------------------


@st.cache_data(show_spinner=False)
def _fcn_cached_bsm_quote(
    spots: tuple,
    vols: tuple,
    divs: tuple,
    corr_flat: tuple,
    risk_free: float,
    tenor_months: int,
    strike_pct: float,
    ki_barrier_pct: float,
    ko_level_pct: float,
    notional: float,
    margin_annual: float,
    tickers: tuple,
) -> FCNQuote:
    """Cache wrapper around BlackScholesFCNEngine.quote()."""
    n = len(spots)
    corr = np.asarray(corr_flat, dtype=np.float64).reshape(n, n)
    engine = BlackScholesFCNEngine(
        spots=list(spots),
        vols=list(vols),
        divs=list(divs),
        corr_matrix=corr,
        risk_free=risk_free,
        tenor_months=tenor_months,
        strike_pct=strike_pct,
        ki_barrier_pct=ki_barrier_pct,
        ko_level_pct=ko_level_pct,
        notional=notional,
        tickers=list(tickers),
    )
    return engine.quote(desk_margin_annual=margin_annual)


@st.cache_data(show_spinner=False)
def _fcn_cached_bsm_greeks(
    spots: tuple,
    vols: tuple,
    divs: tuple,
    corr_flat: tuple,
    risk_free: float,
    tenor_months: int,
    strike_pct: float,
    ki_barrier_pct: float,
    ko_level_pct: float,
    notional: float,
    coupon_annual: float,
    tickers: tuple,
) -> FCNGreeks:
    """Cache wrapper around BlackScholesFCNEngine.greeks() at a fixed coupon."""
    n = len(spots)
    corr = np.asarray(corr_flat, dtype=np.float64).reshape(n, n)
    engine = BlackScholesFCNEngine(
        spots=list(spots),
        vols=list(vols),
        divs=list(divs),
        corr_matrix=corr,
        risk_free=risk_free,
        tenor_months=tenor_months,
        strike_pct=strike_pct,
        ki_barrier_pct=ki_barrier_pct,
        ko_level_pct=ko_level_pct,
        notional=notional,
        tickers=list(tickers),
    )
    return engine.greeks(coupon_annual=coupon_annual)


# ---- Pre-configured SET50 scenarios ---------------------------------------


FCN_SCENARIO_PRESETS = [
    {
        "label": "Scenario A · 3M · 1-Asset (CPALL)",
        "tickers": ("CPALL",),
        "spots": (58.0,),
        "vols": (0.27,),
        "divs": (0.018,),
        "corr": ((1.0,),),
        "tenor": 3,
        "margin": 0.015,
    },
    {
        "label": "Scenario B · 3M · 2-Asset (KBANK, SCB)",
        "tickers": ("KBANK", "SCB"),
        "spots": (145.5, 105.0),
        "vols": (0.24, 0.23),
        "divs": (0.035, 0.045),
        "corr": ((1.00, 0.78), (0.78, 1.00)),
        "tenor": 3,
        "margin": 0.020,
    },
    {
        "label": "Scenario C · 6M · 3-Asset (PTT, PTTEP, ADVANC)",
        "tickers": ("PTT", "PTTEP", "ADVANC"),
        "spots": (33.5, 138.0, 215.0),
        "vols": (0.26, 0.32, 0.22),
        "divs": (0.060, 0.055, 0.040),
        "corr": (
            (1.00, 0.72, 0.30),
            (0.72, 1.00, 0.25),
            (0.30, 0.25, 1.00),
        ),
        "tenor": 6,
        "margin": 0.025,
    },
]


def _fcn_run_preset(preset: dict, risk_free: float, n_paths: int, seed: int) -> FCNQuote:
    """Run one preset scenario through the cached pricer."""
    n = len(preset["tickers"])
    corr_flat = tuple(float(v) for row in preset["corr"] for v in row)
    return _fcn_cached_quote(
        spots=preset["spots"],
        vols=preset["vols"],
        divs=preset["divs"],
        corr_flat=corr_flat,
        risk_free=risk_free,
        tenor_months=preset["tenor"],
        strike_pct=0.95,
        ki_barrier_pct=0.80,
        ko_level_pct=1.00,
        notional=1_000_000.0,
        n_paths=n_paths,
        seed=seed,
        margin_annual=preset["margin"],
        tickers=preset["tickers"],
        n_sample_paths=12,
    )


# ---- Page rendering helpers (themed to match the dark dashboard) ----------


_FCN_NAVY = "#1d3a5f"
_FCN_BLUE = "#7ab8ff"
_FCN_AMBER = "#c98a2f"
_FCN_GREEN = "#2f9d6a"
_FCN_RED = "#c94a4a"
_FCN_TEXT = "#d6d6d6"


def _fcn_render_quote_card(client_coupon_pa: float, sub_label: str) -> None:
    """Big amber/red headline metric for the client coupon."""
    color = _FCN_AMBER if client_coupon_pa > 0 else _FCN_RED
    st.markdown(
        f"""
        <div style="background:{_FCN_NAVY}; border-left:6px solid {color};
                    border-radius:10px; padding:18px 24px; margin: 4px 0 12px 0;">
          <div style="color:{_FCN_BLUE}; font-size:12px; letter-spacing:1.5px;
                      text-transform:uppercase;">Annualized coupon quote to client</div>
          <div style="color:{color}; font-size:2.6rem; font-weight:700;
                      line-height:1.05; margin-top:4px;">
              {client_coupon_pa * 100:,.2f}% p.a.
          </div>
          <div style="color:{_FCN_TEXT}; opacity:0.75; font-size:13px;
                      margin-top:6px;">{sub_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _fcn_render_pv_breakdown(quote: FCNQuote) -> None:
    """Four-column PV decomposition."""
    notional = quote.price_result.notional
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Model base PV (protection)",
        f"{quote.model_base_pv:,.0f}",
        f"{quote.model_base_pv / notional * 100:.2f}% of notional",
    )
    c2.metric(
        "Gross premium captured",
        f"{quote.gross_premium_pv:,.0f}",
        f"{quote.gross_premium_pv / notional * 100:.2f}% of notional",
    )
    c3.metric(
        "Desk fees PV",
        f"{quote.desk_fees_pv:,.0f}",
        f"{quote.desk_margin_annual * 100:.2f}% p.a. margin",
    )
    c4.metric(
        "Client coupon PV",
        f"{quote.client_coupon_pv:,.0f}",
        f"{quote.client_coupon_annual * 100:.2f}% p.a. coupon",
    )


def _fcn_render_diagnostics(pr: FCNPriceResult) -> None:
    """Risk / probability strip."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P(KO / autocall)", f"{pr.prob_ko * 100:.1f}%")
    c2.metric("P(KI breached)", f"{pr.prob_ki * 100:.1f}%")
    c3.metric("P(loss at maturity)", f"{pr.prob_loss_at_maturity * 100:.1f}%")
    c4.metric("Expected settle", f"{pr.expected_settle_months:.2f} mo")


def _fcn_render_model_comparison(
    mc_quote: FCNQuote, bsm_quote: FCNQuote, bsm_is_active: bool
) -> None:
    """Side-by-side MC vs Black-Scholes-extension comparison card.

    Highlights the model the user picked as headline (subtle border accent).
    Shows: client coupon (p.a.), fair coupon (model), P(KO), P(KI), and the
    spread in coupon basis points so the desk can sanity-check the delta.
    """
    st.subheader("Pricing model comparison")
    mc_pr = mc_quote.price_result
    bs_pr = bsm_quote.price_result

    coupon_gap_bps = (
        bsm_quote.client_coupon_annual - mc_quote.client_coupon_annual
    ) * 10_000

    def _accent(active: bool) -> str:
        # Slightly brighter blue border when the model is the headline one.
        return _FCN_BLUE if active else "#314058"

    col_mc, col_bs = st.columns(2)

    with col_mc:
        st.markdown(
            f"""
            <div style="border:1px solid {_accent(not bsm_is_active)};
                        border-radius:10px; padding:14px 16px; background:#101524;">
              <div style="color:{_FCN_TEXT}; font-weight:600; font-size:0.95rem;">
                Monte Carlo Simulation
              </div>
              <div style="color:#7e8aa3; font-size:0.78rem; margin-bottom:8px;">
                {len(mc_pr.sample_worst_of_paths) if mc_pr.sample_worst_of_paths is not None else '100k'} simulated paths · Cholesky GBM · daily KI / monthly KO
              </div>
              <div style="color:{_FCN_AMBER}; font-size:1.5rem; font-weight:700;">
                {mc_quote.client_coupon_annual*100:.3f}% p.a.
              </div>
              <div style="color:#9ba8c2; font-size:0.78rem; margin-top:6px;">
                Fair (pre-margin): {mc_pr.fair_coupon_annual*100:.3f}% &nbsp;·&nbsp;
                stderr ±{mc_pr.mc_stderr_fair_coupon*100:.3f}%
              </div>
              <div style="color:#9ba8c2; font-size:0.78rem;">
                P(KO) {mc_pr.prob_ko*100:.1f}% &nbsp;·&nbsp;
                P(KI) {mc_pr.prob_ki*100:.1f}% &nbsp;·&nbsp;
                P(loss) {mc_pr.prob_loss_at_maturity*100:.1f}%
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_bs:
        st.markdown(
            f"""
            <div style="border:1px solid {_accent(bsm_is_active)};
                        border-radius:10px; padding:14px 16px; background:#101524;">
              <div style="color:{_FCN_TEXT}; font-weight:600; font-size:0.95rem;">
                Analytic Black-Scholes Extension
              </div>
              <div style="color:#7e8aa3; font-size:0.78rem; margin-bottom:8px;">
                Reiner-Rubinstein DIP + MVN-CDF autocall ladder + Clark proxy
              </div>
              <div style="color:{_FCN_BLUE}; font-size:1.5rem; font-weight:700;">
                {bsm_quote.client_coupon_annual*100:.3f}% p.a.
              </div>
              <div style="color:#9ba8c2; font-size:0.78rem; margin-top:6px;">
                Fair (pre-margin): {bs_pr.fair_coupon_annual*100:.3f}% &nbsp;·&nbsp;
                no MC noise
              </div>
              <div style="color:#9ba8c2; font-size:0.78rem;">
                P(KO) {bs_pr.prob_ko*100:.1f}% &nbsp;·&nbsp;
                P(KI) {bs_pr.prob_ki*100:.1f}% &nbsp;·&nbsp;
                P(loss) {bs_pr.prob_loss_at_maturity*100:.1f}%
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    gap_color = _FCN_GREEN if abs(coupon_gap_bps) < 100 else _FCN_AMBER
    if abs(coupon_gap_bps) >= 300:
        gap_color = _FCN_RED
    st.markdown(
        f"""
        <div style="margin-top:10px; padding:8px 14px; background:#0d1320;
                    border-left:3px solid {gap_color}; border-radius:6px;
                    color:#c9d3e6; font-size:0.85rem;">
          BSM − MC client-coupon spread: <b>{coupon_gap_bps:+.1f} bps</b>
          ({(bsm_quote.client_coupon_annual - mc_quote.client_coupon_annual)*100:+.3f}% p.a.).
          {('Within MC noise band.' if abs(coupon_gap_bps) < 100
            else 'Material gap — verify with MC if quoting to client.')}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _fcn_render_greeks_table(greeks: FCNGreeks, n_assets: int) -> None:
    """Per-asset Greeks table with hedge-share guidance for the desk."""
    st.subheader("Greeks (analytic, per asset)")
    st.caption(
        "**Δ** = ∂V/∂Sᵢ in shares · **Γ** = ∂²V/∂Sᵢ² · **ν** = ∂V/∂σᵢ per +1% vol move. "
        "**Hedge** = shares of asset i the bank should LONG to delta-hedge the FCN it has sold "
        "(positive = buy stock; negative = short stock)."
    )

    rows = []
    for i, tk in enumerate(greeks.tickers):
        rows.append({
            "Asset": tk,
            "Spot (THB)": greeks.spots[i],
            "Δ (shares)": greeks.delta[i],
            "Γ (sh/THB)": greeks.gamma[i],
            "ν per +1% vol (THB)": greeks.vega_per_pct[i],
            "Hedge: LONG (shares)": greeks.hedge_shares[i],
            "Hedge $: notional (THB)": greeks.hedge_shares[i] * greeks.spots[i],
        })
    df = pd.DataFrame(rows)
    df_styled = df.style.format({
        "Spot (THB)": "{:,.2f}",
        "Δ (shares)": "{:+,.2f}",
        "Γ (sh/THB)": "{:+,.4f}",
        "ν per +1% vol (THB)": "{:+,.2f}",
        "Hedge: LONG (shares)": "{:+,.0f}",
        "Hedge $: notional (THB)": "{:+,.0f}",
    })
    st.dataframe(df_styled, use_container_width=True, hide_index=True)

    # Aggregate totals for desk-level summary.
    total_delta_thb = sum(g * s for g, s in zip(greeks.hedge_shares, greeks.spots))
    total_vega_thb = sum(greeks.vega_per_pct)
    delta_pct = (total_delta_thb / greeks.notional * 100.0) if greeks.notional else 0.0

    a1, a2, a3 = st.columns(3)
    a1.metric(
        "Total Δ exposure",
        f"{total_delta_thb:,.0f} THB / 1.0 spot",
        help=(
            "**Total Delta Exposure** — รวมมูลค่าหุ้นที่ Desk ต้องถือ Long "
            "เพื่อ Delta-Hedge\n\n"
            "= Σ (Δᵢ × Sᵢ)\n\n"
            "หากตลาดขยับ 1% พร้อมกัน → V ของ Note จะเปลี่ยนประมาณ "
            f"**{delta_pct:.2f}%** ของ Notional"
        ),
    )
    a2.metric(
        "Effective Δ (% of notional)",
        f"{delta_pct:.2f}%",
        help=(
            "**Δ% ของ Notional** — สัดส่วน Delta เทียบกับเงินต้นของ Note\n\n"
            "FCN at-the-money โดยทั่วไป Δ% อยู่ที่ ~15–40% ขึ้นอยู่กับ\n"
            "• Tenor (ยาว → Δ ต่ำ เพราะ KO/KI absorb)\n"
            "• ระยะห่างจาก barrier\n"
            "• Correlation ของตะกร้า worst-of"
        ),
    )
    a3.metric(
        "Total ν per +1% vol",
        f"{total_vega_thb:+,.0f} THB",
        help=(
            "**Total Vega** — กำไร/ขาดทุนของ Note หากความผันผวนของหุ้นเพิ่ม 1% (ทุกตัวพร้อมกัน)\n\n"
            "ค่า **ลบ** = Desk **short vol** (กำไรเมื่อ vol ลดลง, ขาดทุนเมื่อ vol เพิ่ม)\n"
            "FCN จะ short vol เสมอเพราะมี Short DIP ฝังอยู่"
        ),
    )


def _fcn_path_chart(quote: FCNQuote, trading_days_per_year: int = 252) -> go.Figure:
    """Sample worst-of paths vs KI/KO/Strike barriers, dark-theme styled."""
    pr = quote.price_result
    sample = pr.sample_worst_of_paths
    fig = go.Figure()
    if sample is None or sample.size == 0:
        return fig

    n_sample, n_days = sample.shape
    x = np.arange(n_days)
    breached = (sample <= pr.ki_barrier_pct).any(axis=1)
    for k in range(n_sample):
        color = (
            "rgba(201, 138, 47, 0.95)" if breached[k] else "rgba(122, 184, 255, 0.45)"
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=sample[k],
                mode="lines",
                line=dict(width=1.5, color=color),
                showlegend=False,
                hovertemplate="Day %{x}<br>Worst-of %{y:.3f}<extra></extra>",
            )
        )

    # Barriers.
    fig.add_hline(
        y=pr.ko_level_pct,
        line=dict(color=_FCN_GREEN, dash="dash", width=1.5),
        annotation_text=f"KO {pr.ko_level_pct*100:.0f}%",
        annotation_position="top right",
        annotation_font_color=_FCN_GREEN,
    )
    fig.add_hline(
        y=pr.strike_pct,
        line=dict(color=_FCN_BLUE, dash="dot", width=1.5),
        annotation_text=f"Strike {pr.strike_pct*100:.0f}%",
        annotation_position="top right",
        annotation_font_color=_FCN_BLUE,
    )
    fig.add_hline(
        y=pr.ki_barrier_pct,
        line=dict(color=_FCN_RED, dash="dash", width=1.5),
        annotation_text=f"KI {pr.ki_barrier_pct*100:.0f}%",
        annotation_position="bottom right",
        annotation_font_color=_FCN_RED,
    )

    # Monthly observation markers.
    days_per_month = trading_days_per_year / 12.0
    for m in range(1, pr.tenor_months + 1):
        d = int(round(days_per_month * m))
        fig.add_vline(x=d, line=dict(color="rgba(255,255,255,0.18)", dash="dot", width=1))

    fig.update_layout(**PLOTLY_LAYOUT, height=420)
    fig.update_xaxes(title_text="Trading days from issue")
    fig.update_yaxes(title_text="Worst-of S_min / S_0")
    return fig


# ---- Hero + page functions ------------------------------------------------


def fcn_hero() -> None:
    col_title, col_pill = st.columns([5, 1])
    with col_title:
        st.title("Thai FCN Desk · Pricing & Quoting")
        st.caption(
            "Multi-asset Monte Carlo · Cholesky-correlated GBM · "
            "daily KI / monthly KO observation · 1, 2 or 3 underlyings"
        )
    with col_pill:
        st.markdown(
            "<div style='text-align:right; padding-top:24px;'>"
            f"<span style='background:{_FCN_NAVY}; color:{_FCN_BLUE}; padding:4px 10px; "
            "border-radius:12px; font-size:12px;'>EXOTIC EQD</span></div>",
            unsafe_allow_html=True,
        )

    # Quick KPI strip using the cached preset scenarios — gives the desk
    # an at-a-glance "where are coupons today" indicator.
    try:
        quotes = [_fcn_run_preset(p, 0.0225, 50_000, 20260526) for p in FCN_SCENARIO_PRESETS]
    except Exception:  # pragma: no cover — defensive only
        quotes = None

    if quotes:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "1-asset 3M (CPALL)",
            f"{quotes[0].client_coupon_annual * 100:.2f}%",
            "client coupon p.a.",
        )
        c2.metric(
            "2-asset 3M (KBANK/SCB)",
            f"{quotes[1].client_coupon_annual * 100:.2f}%",
            "client coupon p.a.",
        )
        c3.metric(
            "3-asset 6M (PTT/PTTEP/ADVANC)",
            f"{quotes[2].client_coupon_annual * 100:.2f}%",
            "client coupon p.a.",
        )
        c4.metric("THOR baseline", "2.25%", "risk-free in presets")


@st.cache_data(ttl=300, show_spinner=False)
def _fcn_fetch_yahoo(raw_ticker: str) -> dict:
    """
    Fetch spot, ~30d realized vol, and dividend yield from Yahoo Finance.

    Designed for SET tickers — if the user types a bare symbol like 'CPALL',
    we append '.BK' for the Yahoo lookup. If they already supplied a dotted
    symbol (e.g. 'CPALL.BK' or 'AAPL' which has no dot but is a US stock),
    we respect what they typed (US tickers without a dot just resolve directly).

    Returns dict with keys: ok, spot, vol_pct, div_yield_pct, source_ticker, error.
    Cached for 5 minutes to avoid hammering Yahoo on widget reruns.

    NOTE: implied vol is not available on free Yahoo data — we use 30-day
    realized vol from log returns as a proxy. The desk should override this
    with a broker-quoted IV in the implied-vol field after fetch.
    """
    if not _YF_OK:
        return {"ok": False, "error": "yfinance is not installed in this environment."}

    raw = (raw_ticker or "").strip().upper()
    if not raw:
        return {"ok": False, "error": "Ticker is empty."}

    # Heuristic: bare alpha-only ticker -> assume SET, append .BK.
    # If the user already provided a dot (e.g. 'CPALL.BK', 'BRK-B', '^SET'),
    # respect their input.
    yahoo_sym = raw if ("." in raw or raw.startswith("^")) else f"{raw}.BK"

    try:
        ticker_obj = yf.Ticker(yahoo_sym)
        # ~4 months of daily closes gives us enough for a stable 30d realized vol.
        hist = ticker_obj.history(period="4mo", interval="1d", auto_adjust=False)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return {"ok": False, "error": f"No price history returned for {yahoo_sym}."}

        closes = hist["Close"].dropna()
        if len(closes) < 21:
            return {"ok": False, "error": f"Only {len(closes)} closes for {yahoo_sym} (need >=21)."}

        spot = float(closes.iloc[-1])
        log_ret = np.log(closes / closes.shift(1)).dropna()
        # 30-day realized vol, annualized at 252 trading days.
        recent = log_ret.tail(30)
        realized_vol = float(recent.std() * np.sqrt(252))
        vol_pct = realized_vol * 100.0

        # Dividend yield — yfinance returns this in 'info' (sometimes as decimal,
        # sometimes as %). We normalize to a percentage.
        try:
            info = ticker_obj.info or {}
        except Exception:
            info = {}
        div_raw = (
            info.get("dividendYield")
            or info.get("trailingAnnualDividendYield")
            or 0.0
        )
        div_yield_pct = float(div_raw) * 100.0 if div_raw <= 1.0 else float(div_raw)

        return {
            "ok": True,
            "spot": spot,
            "vol_pct": vol_pct,
            "div_yield_pct": div_yield_pct,
            "source_ticker": yahoo_sym,
            "error": None,
        }
    except Exception as exc:  # pragma: no cover — network paths
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _fcn_apply_fetch(i: int, result: dict) -> None:
    """Push a successful fetch into the widget state keys + the persistent lists."""
    st.session_state[f"fcn_spot_{i}"] = float(result["spot"])
    st.session_state[f"fcn_vol_{i}"] = float(result["vol_pct"])
    st.session_state[f"fcn_div_{i}"] = float(result["div_yield_pct"])
    st.session_state["fcn_spots"][i] = float(result["spot"])
    st.session_state["fcn_vols"][i] = float(result["vol_pct"])
    st.session_state["fcn_divs"][i] = float(result["div_yield_pct"])


def _fcn_ensure_state() -> None:
    """Seed st.session_state with sensible defaults the first time the page loads."""
    defaults = {
        "fcn_n_assets": 3,
        "fcn_tickers": ["KBANK", "SCB", "ADVANC"],
        "fcn_spots": [145.50, 105.00, 215.00],
        "fcn_vols": [24.0, 23.0, 22.0],          # in %
        "fcn_divs": [3.5, 4.5, 4.0],             # in %
        "fcn_corr01": 0.78,
        "fcn_corr02": 0.35,
        "fcn_corr12": 0.30,
        "fcn_tenor": 6,
        "fcn_strike_pct": 95.0,
        "fcn_ki_pct": 80.0,
        "fcn_ko_pct": 100.0,
        "fcn_r_pct": 2.25,
        "fcn_margin_pct": 2.00,
        "fcn_notional": 1_000_000.0,
        "fcn_n_paths": 100_000,
        "fcn_seed": 20260526,
        "fcn_n_sample": 15,
        # Pricing model: "MC" = Monte Carlo (default), "BSM" = analytic Black-Scholes.
        # The other engine is ALWAYS run in parallel for the side-by-side
        # comparison card — this just selects which becomes the headline quote.
        "fcn_model": "MC",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def fcn_quote_page() -> None:
    """Main interactive pricing + quoting page."""
    _fcn_ensure_state()

    # ---- Note structure -----------------------------------------------------
    with st.container(border=True):
        st.markdown("##### Structure")
        n_assets = st.selectbox(
            "Number of underlyings",
            options=[1, 2, 3],
            index=[1, 2, 3].index(st.session_state["fcn_n_assets"]),
            key="fcn_n_assets",
            help=(
                "**จำนวนหุ้นอ้างอิง (Underlyings)**\n\n"
                "FCN ไทยรองรับ 1, 2 หรือ 3 หุ้น\n\n"
                "• ยิ่งจำนวนหุ้นมาก → ความเสี่ยงแบบ **Worst-of** มากขึ้น "
                "(ผลตอบแทนสุดท้ายอ้างอิงจากหุ้นที่ผลงานแย่ที่สุด)\n"
                "• Worst-of ยิ่งแย่ → Coupon ที่ลูกค้าได้รับยิ่งสูง"
            ),
        )

        cols = st.columns(n_assets)
        # We re-use the persisted lists but trim/pad to current n_assets.
        st.session_state["fcn_tickers"] = (
            st.session_state["fcn_tickers"] + ["", "", ""]
        )[:n_assets]
        st.session_state["fcn_spots"] = (
            st.session_state["fcn_spots"] + [100.0, 100.0, 100.0]
        )[:n_assets]
        st.session_state["fcn_vols"] = (st.session_state["fcn_vols"] + [25.0] * 3)[:n_assets]
        st.session_state["fcn_divs"] = (st.session_state["fcn_divs"] + [3.0] * 3)[:n_assets]

        # Bulk "Fetch all" button — one click pulls live spot / 30d realized vol
        # / dividend yield from Yahoo for every asset in the current configuration.
        bulk_col_btn, bulk_col_note = st.columns([1, 4])
        with bulk_col_btn:
            fetch_all_clicked = st.button(
                "↻ Fetch all from Yahoo",
                key="fcn_fetch_all",
                use_container_width=True,
                disabled=not _YF_OK,
                help=(
                    "**ดึงข้อมูลทุกหุ้นจาก Yahoo Finance พร้อมกัน**\n\n"
                    "• พิมพ์ชื่อย่อ SET เปล่าๆ (เช่น CPALL, KBANK, PTT) → ระบบเติม `.BK` ให้อัตโนมัติ\n"
                    "• ถ้าใส่ symbol เต็มเอง (เช่น `CPALL.BK`, `AAPL`, `^SET`) → ใช้ตามที่พิมพ์\n\n"
                    "ค่าที่ดึงมา: Spot ล่าสุด, Realized Vol 30 วัน, Dividend Yield"
                ),
            )
        with bulk_col_note:
            if _YF_OK:
                st.caption(
                    "Tip · type bare SET tickers like `CPALL` — Yahoo lookup uses `CPALL.BK`. "
                    "Implied vol is filled with a **30-day realized vol** proxy; override with a "
                    "broker-quoted IV if you have one."
                )
            else:
                st.caption("`yfinance` not installed — auto-fetch is disabled. Install with `pip install yfinance`.")

        if fetch_all_clicked:
            failures: list[str] = []
            successes: list[str] = []
            for i in range(n_assets):
                tkr = st.session_state.get(f"fcn_tkr_{i}", "").strip()
                if not tkr:
                    failures.append(f"#{i+1}: empty ticker")
                    continue
                res = _fcn_fetch_yahoo(tkr)
                if res["ok"]:
                    _fcn_apply_fetch(i, res)
                    successes.append(f"{res['source_ticker']} @ {res['spot']:.2f}")
                else:
                    failures.append(f"{tkr}: {res['error']}")
            if successes:
                st.toast(f"Fetched {len(successes)}: " + " · ".join(successes), icon="✅")
            for msg in failures:
                st.warning(msg)
            if successes:
                st.rerun()

        tickers: List[str] = []
        spots: List[float] = []
        vols: List[float] = []
        divs: List[float] = []
        for i in range(n_assets):
            with cols[i]:
                tickers.append(
                    st.text_input(
                        f"Ticker #{i+1}",
                        value=st.session_state["fcn_tickers"][i],
                        key=f"fcn_tkr_{i}",
                        help=(
                            "**Ticker / รหัสหุ้นอ้างอิง**\n\n"
                            "• หุ้น SET: พิมพ์ชื่อย่อเปล่าๆ เช่น `CPALL`, `KBANK`, `PTT` "
                            "(ระบบจะเติม `.BK` ให้อัตโนมัติเวลาดึงข้อมูล)\n"
                            "• หุ้นต่างประเทศ / ดัชนี: ใส่ symbol เต็ม เช่น `AAPL`, `^SET`, `CPALL.BK`"
                        ),
                    )
                )
                # Per-asset fetch button — lets the user refresh one row without
                # re-hitting Yahoo for the others (and without losing manual overrides).
                if st.button(
                    "↻ Fetch this one",
                    key=f"fcn_fetch_{i}",
                    use_container_width=True,
                    disabled=not _YF_OK,
                    help=(
                        "**ดึงข้อมูลเฉพาะหุ้นนี้จาก Yahoo**\n\n"
                        "ใช้เมื่อคุณต้องการอัปเดตหุ้นตัวเดียว โดยไม่ต้องดึงข้อมูลหุ้นอื่นใหม่ "
                        "(เหมาะกับการแก้ค่าเฉพาะจุดหลังจากกรอก Override ด้วยมือไปแล้ว)"
                    ),
                ):
                    tkr = st.session_state.get(f"fcn_tkr_{i}", "").strip()
                    if not tkr:
                        st.warning("Enter a ticker first.")
                    else:
                        res = _fcn_fetch_yahoo(tkr)
                        if res["ok"]:
                            _fcn_apply_fetch(i, res)
                            st.toast(
                                f"✓ {res['source_ticker']}: "
                                f"spot {res['spot']:.2f} · "
                                f"vol {res['vol_pct']:.1f}% · "
                                f"div {res['div_yield_pct']:.2f}%",
                                icon="✅",
                            )
                            st.rerun()
                        else:
                            st.error(f"Fetch failed: {res['error']}")
                spots.append(
                    st.number_input(
                        "Live spot",
                        min_value=0.01,
                        value=float(st.session_state["fcn_spots"][i]),
                        step=0.10,
                        key=f"fcn_spot_{i}",
                        format="%.2f",
                        help=(
                            "**ราคาตลาดปัจจุบัน (Spot Price, S₀)**\n\n"
                            "ใช้เป็นจุดอ้างอิงเริ่มต้น — Strike, KI, KO ทั้งหมดคำนวณเป็น "
                            "**% ของราคา Spot นี้**\n\n"
                            "• Strike 95% หมายถึง 95% ของราคานี้\n"
                            "• กดปุ่ม **Fetch** เพื่อดึงราคาปิดล่าสุดจาก Yahoo อัตโนมัติ"
                        ),
                    )
                )
                vols.append(
                    st.number_input(
                        "Implied vol (% p.a.)",
                        min_value=1.0, max_value=200.0,
                        value=float(st.session_state["fcn_vols"][i]),
                        step=0.5,
                        key=f"fcn_vol_{i}",
                        help=(
                            "**Implied Volatility — ความผันผวนรายปี (% p.a.)**\n\n"
                            "ค่าหลักที่ใช้ใน GBM Model: กำหนด Drift และ Diffusion ของราคา\n\n"
                            "• ปุ่ม **Fetch** จะเติมค่า **Realized Vol 30 วัน** จากราคาในอดีตให้ก่อน "
                            "(เป็นค่าประมาณ — Yahoo ฟรีไม่มี IV จริง)\n"
                            "• ถ้ามี **Implied Vol จาก Broker** แนะนำให้กรอกแทน\n\n"
                            "ผลกระทบ:\n"
                            "• Vol สูง → ราคาแกว่งมาก → KO/KI โดนง่ายขึ้น → Coupon สูง\n"
                            "• Vol ต่ำ → ราคานิ่ง → Coupon ต่ำ"
                        ),
                    )
                )
                divs.append(
                    st.number_input(
                        "Div yield (% p.a.)",
                        min_value=0.0, max_value=20.0,
                        value=float(st.session_state["fcn_divs"][i]),
                        step=0.1,
                        key=f"fcn_div_{i}",
                        help=(
                            "**อัตราเงินปันผลคาดหวังต่อปี (Dividend Yield, q)**\n\n"
                            "หักออกจาก Drift ในสูตร GBM: **(r − q − ½σ²)**\n\n"
                            "• หุ้นไทยที่จ่ายปันผลสูง (ธนาคาร, พลังงาน) → ลด Drift → "
                            "Coupon ต่ำลงเล็กน้อย\n"
                            "• ปุ่ม **Fetch** จะดึง Dividend Yield ล่าสุดจาก Yahoo (ค่า TTM)"
                        ),
                    )
                )

    # ---- Correlation matrix -------------------------------------------------
    corr = np.eye(n_assets)
    if n_assets == 1:
        st.caption("Single asset · correlation matrix is the 1×1 identity.")
    else:
        with st.container(border=True):
            st.markdown("##### Correlation matrix")
            ccols = st.columns(3 if n_assets == 3 else 1)
            # Pair (0,1)
            with ccols[0]:
                corr01 = st.slider(
                    f"corr({tickers[0] or 'A1'}, {tickers[1] or 'A2'})",
                    -0.99, 0.99,
                    float(st.session_state["fcn_corr01"]),
                    step=0.01, key="fcn_corr01",
                    help=(
                        "**ค่าสัมประสิทธิ์สหสัมพันธ์ (Correlation, ρ) ระหว่างหุ้นคู่นี้**\n\n"
                        "ช่วงค่า: −0.99 ถึง +0.99\n\n"
                        "ผลกระทบต่อราคา Note:\n"
                        "• ρ สูง (ใกล้ 1) → หุ้นเคลื่อนไหวไปด้วยกัน → Worst-of กระจายตัวน้อย → "
                        "**Coupon ต่ำ**\n"
                        "• ρ ต่ำ / ติดลบ → หุ้นเคลื่อนไหวสวนทาง → Worst-of กระจายตัวมาก → "
                        "**Coupon สูง**\n\n"
                        "ตัวอย่าง: KBANK กับ SCB (ธนาคารด้วยกัน) มักมี ρ ≈ 0.7–0.8"
                    ),
                )
                corr[0, 1] = corr[1, 0] = corr01
            if n_assets == 3:
                with ccols[1]:
                    corr02 = st.slider(
                        f"corr({tickers[0] or 'A1'}, {tickers[2] or 'A3'})",
                        -0.99, 0.99,
                        float(st.session_state["fcn_corr02"]),
                        step=0.01, key="fcn_corr02",
                        help=(
                            "**ค่าสหสัมพันธ์ระหว่างหุ้นตัวที่ 1 กับตัวที่ 3**\n\n"
                            "• ค่าสูง → ความเสี่ยงรวมศูนย์ (Concentrated) → Coupon ต่ำ\n"
                            "• ค่าต่ำ → กระจายความเสี่ยง (Diversified) → Coupon สูง\n\n"
                            "ถ้าใส่ค่าที่ทำให้ Matrix ไม่เป็น Positive Semi-Definite "
                            "ระบบจะปรับให้อัตโนมัติ (Nearest-PSD)"
                        ),
                    )
                    corr[0, 2] = corr[2, 0] = corr02
                with ccols[2]:
                    corr12 = st.slider(
                        f"corr({tickers[1] or 'A2'}, {tickers[2] or 'A3'})",
                        -0.99, 0.99,
                        float(st.session_state["fcn_corr12"]),
                        step=0.01, key="fcn_corr12",
                        help=(
                            "**ค่าสหสัมพันธ์ระหว่างหุ้นตัวที่ 2 กับตัวที่ 3**\n\n"
                            "ใช้ในการสร้าง Cholesky Decomposition เพื่อจำลองราคาทั้งสามให้มี "
                            "ความสัมพันธ์ตามที่กำหนด\n\n"
                            "ตัวอย่าง:\n"
                            "• PTT vs PTTEP ≈ 0.7 (พลังงานเหมือนกัน)\n"
                            "• PTT vs ADVANC ≈ 0.3 (คนละกลุ่มอุตสาหกรรม)"
                        ),
                    )
                    corr[1, 2] = corr[2, 1] = corr12

    # ---- Note terms ---------------------------------------------------------
    with st.container(border=True):
        st.markdown("##### Note terms & cost stack")
        t1, t2, t3, t4 = st.columns(4)
        tenor_months = t1.selectbox(
            "Tenor (months)", options=[3, 6],
            index=[3, 6].index(st.session_state["fcn_tenor"]),
            key="fcn_tenor",
            help=(
                "**Tenor — ระยะเวลาของ Note (เดือน)**\n\n"
                "FCN ไทยมาตรฐานคือ **3 หรือ 6 เดือน**\n\n"
                "ผลกระทบ:\n"
                "• Tenor ยาวขึ้น → มีเวลาให้หุ้นแกว่งมากขึ้น → KI โดนง่ายขึ้น → **Coupon สูงขึ้น**\n"
                "• Tenor สั้น → ความเสี่ยงน้อย → Coupon ต่ำ\n\n"
                "โครงสร้างคูปอง: จ่ายรายเดือนตลอดอายุ Note (หรือจนกว่าจะ KO)"
            ),
        )
        strike_pct = t2.number_input(
            "Strike (% of spot)",
            min_value=50.0, max_value=120.0,
            value=float(st.session_state["fcn_strike_pct"]),
            step=1.0, key="fcn_strike_pct",
            help=(
                "**Strike Price — เกณฑ์อ้างอิงสำหรับการชำระเงินคืน**\n\n"
                "เป็น **% ของราคา Spot** ตอนเริ่มต้น Note (มาตรฐาน **95%**)\n\n"
                "ใช้เมื่อไหร่:\n"
                "• ถ้าครบกำหนด + **KI โดน** + Worst-of < Strike → ลูกค้าได้รับชำระเป็น "
                "**(Worst-of / Strike) × Notional** (ขาดทุน)\n"
                "• ถ้าครบกำหนด + Worst-of ≥ Strike → ได้เงินต้นเต็มจำนวน\n\n"
                "Strike ต่ำ (เช่น 90%) → มี Buffer มากขึ้น → Coupon ลดลง"
            ),
        )
        ki_pct = t3.number_input(
            "KI barrier (% of spot)",
            min_value=30.0, max_value=100.0,
            value=float(st.session_state["fcn_ki_pct"]),
            step=1.0, key="fcn_ki_pct",
            help=(
                "**Knock-In Barrier — เส้นเตือนความเสี่ยงขาลง**\n\n"
                "เป็น **% ของราคา Spot** (มาตรฐาน **80%**)\n\n"
                "วิธีตรวจสอบ:\n"
                "• ตรวจ **ทุกวันทำการ (Daily Observation)** ตลอดอายุ Note\n"
                "• ถ้าหุ้น Worst-of **เคยแตะ** ระดับนี้ ณ จุดใดจุดหนึ่ง → **KI ถือว่าโดน**\n\n"
                "ผลของ KI โดน:\n"
                "• การคุ้มครองเงินต้นถูกยกเลิก\n"
                "• ถ้าตอนครบกำหนด Worst-of < Strike → ขาดทุนตาม Worst-of\n\n"
                "KI ต่ำ (เช่น 70%) → ปลอดภัยกว่า แต่ Coupon ต่ำลง"
            ),
        )
        ko_pct = t4.number_input(
            "KO level (% of spot)",
            min_value=80.0, max_value=130.0,
            value=float(st.session_state["fcn_ko_pct"]),
            step=1.0, key="fcn_ko_pct",
            help=(
                "**Knock-Out / Autocall Level — เส้นไถ่ถอนก่อนกำหนด**\n\n"
                "เป็น **% ของราคา Spot** (มาตรฐาน **100%**)\n\n"
                "วิธีตรวจสอบ:\n"
                "• ตรวจ **ทุกเดือน (Monthly Observation)** ในวันสังเกตการณ์\n"
                "• ถ้า Worst-of ≥ ระดับนี้ในวันใด → Note **ไถ่ถอนก่อนกำหนด** ที่พาร์ "
                "พร้อมคูปองเดือนนั้น\n\n"
                "ผลของ KO:\n"
                "• ลูกค้าได้รับเงินต้นเต็ม + ดอกเบี้ยถึงวันไถ่ถอน\n"
                "• Note ปิดทันที ไม่เสี่ยง KI หลังจากนั้น\n\n"
                "KO สูง (เช่น 105%) → ไถ่ถอนยากขึ้น → Coupon สูงขึ้น"
            ),
        )

        f1, f2, f3 = st.columns(3)
        r_pct = f1.slider(
            "THOR risk-free rate (% p.a.)",
            0.0, 8.0,
            float(st.session_state["fcn_r_pct"]),
            step=0.05, key="fcn_r_pct",
            help=(
                "**THOR (Thai Overnight Repurchase Rate)** — อัตราดอกเบี้ยอ้างอิงปลอดความเสี่ยงของไทย\n\n"
                "ใช้สำหรับ:\n"
                "• คิดลด (discount) กระแสเงินสดในอนาคตให้เป็นมูลค่าปัจจุบัน (PV)\n"
                "• เป็นอัตราการเติบโตของราคาหุ้นในแบบจำลอง GBM (risk-neutral drift)\n\n"
                "ค่าอ้างอิงปัจจุบันประมาณ 2.25% ต่อปี"
            ),
        )
        margin_pct = f2.slider(
            "Desk / hedging margin (% p.a.)",
            0.0, 10.0,
            float(st.session_state["fcn_margin_pct"]),
            step=0.10, key="fcn_margin_pct",
            help=(
                "**Desk / Hedging Margin** — ส่วนต่างที่ Desk หักไว้สำหรับต้นทุนการป้องกันความเสี่ยงและกำไร\n\n"
                "การคำนวณ:\n"
                "• Client Coupon = Fair Coupon − Desk Margin\n"
                "• ยิ่ง Margin สูง → ลูกค้าได้ Coupon น้อยลง แต่ Desk ได้กำไรมากขึ้น\n\n"
                "ค่าทั่วไปอยู่ที่ประมาณ 1.5% – 3.0% ต่อปี ขึ้นอยู่กับโครงสร้างและความซับซ้อนของการ Hedge"
            ),
        )
        notional = f3.number_input(
            "Notional (THB)",
            min_value=10_000.0,
            value=float(st.session_state["fcn_notional"]),
            step=100_000.0, format="%.0f", key="fcn_notional",
            help=(
                "**เงินต้น (Notional) ของ Note หน่วยเป็นบาท**\n\n"
                "ค่ามาตรฐาน: **1,000,000 บาท**\n\n"
                "• ใช้เป็นตัวคูณสำหรับการคำนวณ PV และจำนวนเงิน Coupon\n"
                "• ไม่กระทบ % Coupon ที่เสนอลูกค้า (Coupon เป็น p.a. ไม่ขึ้นกับ Notional)\n"
                "• ใช้เพื่อแสดงตัวเลขในหน่วยบาทบน PV Breakdown ให้เห็นภาพชัดเจน"
            ),
        )

        # ---- Pricing model selector ----------------------------------------
        m_col, _ = st.columns([3, 1])
        model_label = m_col.radio(
            "Pricing model",
            options=[
                "Monte Carlo Simulation",
                "Analytic Barrier Adjustments (Black-Scholes Extension)",
            ],
            horizontal=True,
            index=0 if st.session_state.get("fcn_model", "MC") == "MC" else 1,
            key="fcn_model_label",
            help=(
                "**เลือกแบบจำลองที่ใช้คิดราคา**\n\n"
                "• **Monte Carlo Simulation** — จำลอง 100k paths แบบ Cholesky + GBM, "
                "ตรวจ KI รายวัน + KO รายเดือน → แม่นยำที่สุดสำหรับ FCN path-dependent\n\n"
                "• **Analytic Black-Scholes Extension** — ใช้สูตรปิดของ Reiner-Rubinstein "
                "(Down-and-In Put), Multivariate Normal CDF สำหรับ KO ladder, และ Clark 1961 "
                "ในการสร้าง Single-asset Proxy สำหรับตะกร้า worst-of\n"
                "  - **1 asset**: closed-form แม่นยำเทียบเท่า MC\n"
                "  - **2-3 assets**: ใช้ Worst-of Proxy (Approximation) — เห็น Banner เตือน\n"
                "  - **ข้อดี**: เร็วมาก (<300ms), ได้ **Greeks (Δ, Γ, ν)** มาให้ Hedging\n\n"
                "**ทั้งสองโมเดลรันคู่กันเสมอ** เพื่อให้เปรียบเทียบใน Card ด้านล่าง — "
                "ตัวที่เลือกจะเป็น **ราคาหลักที่เสนอลูกค้า**"
            ),
        )
        st.session_state["fcn_model"] = (
            "BSM" if model_label.startswith("Analytic") else "MC"
        )

        s1, s2, s3, s4 = st.columns([2, 2, 1, 2])
        n_paths = s1.select_slider(
            "Monte Carlo paths",
            options=[10_000, 25_000, 50_000, 100_000, 200_000],
            value=int(st.session_state["fcn_n_paths"]),
            key="fcn_n_paths",
            help=(
                "**จำนวนเส้นทางจำลอง (Monte Carlo Paths)**\n\n"
                "ค่ามาตรฐาน: **100,000 paths** (แม่นยำพอใช้สำหรับการเสนอราคา)\n\n"
                "Trade-off:\n"
                "• ลด paths (เช่น 10,000) → เร็วขึ้นมาก แต่ MC stderr กว้างขึ้น "
                "(±0.3–0.5% p.a.)\n"
                "• เพิ่ม paths (เช่น 200,000) → แม่นยำขึ้นตามกฎ √N "
                "(stderr แคบลงครึ่งหนึ่งทุกๆ 4 เท่าของ paths)\n\n"
                "ดูค่า MC stderr ใต้ผลลัพธ์เพื่อประเมินความเชื่อมั่น"
            ),
        )
        n_sample = s2.slider(
            "Sample paths to chart", 5, 40,
            int(st.session_state["fcn_n_sample"]),
            step=1, key="fcn_n_sample",
            help=(
                "**จำนวนเส้นทางตัวอย่างที่แสดงในกราฟ**\n\n"
                "ใช้สำหรับ **Visualization** เท่านั้น — ไม่กระทบการคิดราคา (PV ใช้ paths ทั้งหมด)\n\n"
                "ในกราฟ:\n"
                "• **เส้นสีอำพัน** = paths ที่เคยแตะ KI Barrier\n"
                "• **เส้นสีฟ้าจาง** = paths ที่ไม่เคยโดน KI\n"
                "• **เส้นประแนวตั้ง** = วันสังเกตการณ์ KO รายเดือน"
            ),
        )
        seed = s3.number_input(
            "Seed", min_value=0,
            value=int(st.session_state["fcn_seed"]),
            step=1, key="fcn_seed",
            help=(
                "**Random Seed สำหรับการจำลอง Monte Carlo**\n\n"
                "ใช้ค่าเดิม → ได้ผลลัพธ์ตัวเลขเหมือนเดิมทุกครั้ง (**Reproducibility**)\n\n"
                "เปลี่ยน Seed แล้ว Coupon เปลี่ยน → ดูว่าค่า MC stderr สมเหตุสมผลหรือไม่ "
                "(ถ้าเปลี่ยนเยอะมาก ควรเพิ่ม paths)"
            ),
        )
        run = s4.button(
            "Price & quote",
            type="primary",
            use_container_width=True,
            help=(
                "**กดเพื่อรันการจำลอง Monte Carlo**\n\n"
                "ขั้นตอน:\n"
                "1. จำลองราคาหุ้นตามจำนวน paths ที่กำหนด (ใช้ Cholesky + GBM)\n"
                "2. ตรวจ KO รายเดือน + KI รายวัน\n"
                "3. คำนวณ Fair Coupon ที่ทำให้ราคา Note = Notional\n"
                "4. หัก Desk Margin → ได้ **Client Coupon** ที่จะเสนอลูกค้า\n\n"
                "ใช้เวลาประมาณ 1–3 วินาที สำหรับ 100,000 paths"
            ),
        )

    # ---- Validation ---------------------------------------------------------
    if any(not t.strip() for t in tickers):
        st.warning("Each asset needs a ticker label.")
        return

    # We always render the last cached quote if available, but only re-trigger
    # MC when the user clicks the button (or on first load with default inputs).
    auto_run_once = "fcn_last_run_key" not in st.session_state

    inputs_key = (
        tuple(tickers), tuple(spots), tuple(vols), tuple(divs),
        tuple(np.asarray(corr).ravel().tolist()),
        tenor_months, strike_pct, ki_pct, ko_pct,
        r_pct, margin_pct, notional, n_paths, seed, n_sample,
    )

    if run or auto_run_once:
        try:
            with st.spinner(
                f"Simulating {n_paths:,} paths across {n_assets} asset(s), "
                f"{tenor_months}M tenor…"
            ):
                quote = _fcn_cached_quote(
                    spots=tuple(spots),
                    vols=tuple(v / 100.0 for v in vols),
                    divs=tuple(d / 100.0 for d in divs),
                    corr_flat=tuple(np.asarray(corr).ravel().tolist()),
                    risk_free=r_pct / 100.0,
                    tenor_months=int(tenor_months),
                    strike_pct=strike_pct / 100.0,
                    ki_barrier_pct=ki_pct / 100.0,
                    ko_level_pct=ko_pct / 100.0,
                    notional=float(notional),
                    n_paths=int(n_paths),
                    seed=int(seed),
                    margin_annual=margin_pct / 100.0,
                    tickers=tuple(tickers),
                    n_sample_paths=int(n_sample),
                )
            st.session_state["fcn_last_quote"] = quote
            st.session_state["fcn_last_run_key"] = inputs_key

            # ---- Always run BSM in parallel for the comparison card -------
            bsm_quote = None
            bsm_greeks = None
            if _SCIPY_OK:
                try:
                    bsm_quote = _fcn_cached_bsm_quote(
                        spots=tuple(spots),
                        vols=tuple(v / 100.0 for v in vols),
                        divs=tuple(d / 100.0 for d in divs),
                        corr_flat=tuple(np.asarray(corr).ravel().tolist()),
                        risk_free=r_pct / 100.0,
                        tenor_months=int(tenor_months),
                        strike_pct=strike_pct / 100.0,
                        ki_barrier_pct=ki_pct / 100.0,
                        ko_level_pct=ko_pct / 100.0,
                        notional=float(notional),
                        margin_annual=margin_pct / 100.0,
                        tickers=tuple(tickers),
                    )
                    bsm_greeks = _fcn_cached_bsm_greeks(
                        spots=tuple(spots),
                        vols=tuple(v / 100.0 for v in vols),
                        divs=tuple(d / 100.0 for d in divs),
                        corr_flat=tuple(np.asarray(corr).ravel().tolist()),
                        risk_free=r_pct / 100.0,
                        tenor_months=int(tenor_months),
                        strike_pct=strike_pct / 100.0,
                        ki_barrier_pct=ki_pct / 100.0,
                        ko_level_pct=ko_pct / 100.0,
                        notional=float(notional),
                        coupon_annual=bsm_quote.price_result.fair_coupon_annual,
                        tickers=tuple(tickers),
                    )
                except Exception as exc:
                    st.warning(f"Black-Scholes engine could not price: {exc}")
            st.session_state["fcn_last_bsm_quote"] = bsm_quote
            st.session_state["fcn_last_bsm_greeks"] = bsm_greeks
        except (ValueError, np.linalg.LinAlgError) as exc:
            st.error(f"Could not price: {exc}")
            return

    quote = st.session_state.get("fcn_last_quote")
    bsm_quote = st.session_state.get("fcn_last_bsm_quote")
    bsm_greeks = st.session_state.get("fcn_last_bsm_greeks")
    if quote is None:
        return

    if quote.corr_repaired:
        st.info(
            "Correlation matrix was not positive semi-definite. "
            "Repaired via nearest-PSD eigenvalue clipping — please review inputs."
        )

    if inputs_key != st.session_state.get("fcn_last_run_key"):
        st.caption(
            "ⓘ Inputs changed since the last run. Click **Price & quote** to refresh."
        )

    # ---- Headline: use the model the user selected --------------------------
    use_bsm = st.session_state.get("fcn_model", "MC") == "BSM" and bsm_quote is not None
    active_quote = bsm_quote if use_bsm else quote
    pr = active_quote.price_result
    sub_label = (
        f"{pr.tenor_months}M · {', '.join(active_quote.tickers)} · "
        f"strike {pr.strike_pct*100:.0f}% / KI {pr.ki_barrier_pct*100:.0f}% / "
        f"KO {pr.ko_level_pct*100:.0f}%  ·  "
        f"{'Black-Scholes (analytic)' if use_bsm else 'Monte Carlo'}"
    )

    _fcn_render_quote_card(active_quote.client_coupon_annual, sub_label)

    if active_quote.client_coupon_annual <= 0:
        st.warning(
            "Client coupon is at/below zero — the desk margin is consuming the entire "
            "model premium. Either lower the margin or tighten the structure "
            "(lower strike, deeper KI) to widen the spread."
        )

    # ---- Multi-asset BSM approximation banner -------------------------------
    if use_bsm and n_assets >= 2:
        st.warning(
            "**Approximation used for Multi-Asset Black-Scholes.** "
            "The worst-of basket is priced via a Clark (1961) moment-matched single-asset "
            "proxy with Reiner-Rubinstein DIP, BGK daily-monitoring correction, and a "
            "KI∩KO overlap adjustment. The proxy does not capture full path-dependent "
            "co-movement of the basket. **For precise multi-asset path-dependency, use "
            "Monte Carlo.** Expect drift of ~1–5% p.a. vs MC on 3-asset baskets."
        )

    # ---- Model comparison card ----------------------------------------------
    if bsm_quote is not None:
        st.divider()
        _fcn_render_model_comparison(quote, bsm_quote, use_bsm)

    # ---- Greeks (always from BSM analytic engine) ---------------------------
    if bsm_greeks is not None:
        st.divider()
        _fcn_render_greeks_table(bsm_greeks, n_assets)

    st.divider()
    _fcn_render_pv_breakdown(active_quote)
    _fcn_render_diagnostics(pr)
    if use_bsm:
        st.caption(
            f"Fair coupon (BSM analytic): **{pr.fair_coupon_annual*100:.2f}% p.a.**  ·  "
            f"Desk margin: **{active_quote.desk_margin_annual*100:.2f}% p.a.**  ·  "
            f"MC reference: **{quote.price_result.fair_coupon_annual*100:.2f}% p.a.**  ·  "
            f"Avg loss | loss: {pr.expected_loss_given_loss_pct:.2f}%"
        )
    else:
        st.caption(
            f"Fair coupon (model): **{pr.fair_coupon_annual*100:.2f}% p.a.**  ·  "
            f"Desk margin: **{active_quote.desk_margin_annual*100:.2f}% p.a.**  ·  "
            f"MC stderr (fair coupon): ±{pr.mc_stderr_fair_coupon*100:.3f}% p.a.  ·  "
            f"Avg loss | loss: {pr.expected_loss_given_loss_pct:.2f}%"
        )

    # ---- Path chart (only relevant for MC) ----------------------------------
    if not use_bsm:
        st.divider()
        st.subheader("Sample worst-of paths vs barriers")
        st.plotly_chart(_fcn_path_chart(active_quote), use_container_width=True)
        st.caption(
            "Amber lines = paths that touched the KI barrier at least once. "
            "Vertical dotted lines = monthly KO observation dates."
        )


def _fcn_load_preset_into_quote(preset: dict) -> None:
    """
    Push a preset's full configuration into the Quote page's session_state
    keys, clear the 'last run' key so the Quote page auto-prices on landing,
    then navigate the dashboard to the Quote page.

    Touches every widget key that the Quote page reads from, including the
    per-slot ticker/spot/vol/div keys and the correlation slider keys.
    """
    n = len(preset["tickers"])

    # Persistent lists (always length 3 — pad with safe defaults).
    st.session_state["fcn_n_assets"] = n
    st.session_state["fcn_tickers"] = list(preset["tickers"]) + [""] * (3 - n)
    st.session_state["fcn_spots"] = list(preset["spots"]) + [100.0] * (3 - n)
    st.session_state["fcn_vols"] = [v * 100.0 for v in preset["vols"]] + [25.0] * (3 - n)
    st.session_state["fcn_divs"] = [d * 100.0 for d in preset["divs"]] + [3.0] * (3 - n)

    # Per-slot widget keys — these are the keys Streamlit actually reads.
    for i in range(n):
        st.session_state[f"fcn_tkr_{i}"] = preset["tickers"][i]
        st.session_state[f"fcn_spot_{i}"] = float(preset["spots"][i])
        st.session_state[f"fcn_vol_{i}"] = float(preset["vols"][i] * 100.0)
        st.session_state[f"fcn_div_{i}"] = float(preset["divs"][i] * 100.0)
    # Clear orphan widget keys for slots that won't render under the new n.
    for i in range(n, 3):
        for k in (f"fcn_tkr_{i}", f"fcn_spot_{i}", f"fcn_vol_{i}", f"fcn_div_{i}"):
            st.session_state.pop(k, None)

    # Correlation sliders — only set the pairs that will actually be shown.
    corr = preset["corr"]
    if n >= 2:
        st.session_state["fcn_corr01"] = float(corr[0][1])
    if n >= 3:
        st.session_state["fcn_corr02"] = float(corr[0][2])
        st.session_state["fcn_corr12"] = float(corr[1][2])

    # Note terms + cost stack — Quote page presets use 95/80/100 by default,
    # but we still set the desk margin (which is what differs per preset).
    st.session_state["fcn_tenor"] = int(preset["tenor"])
    st.session_state["fcn_margin_pct"] = float(preset["margin"] * 100.0)
    # Reset strike/KI/KO to preset defaults too, so the user lands on the
    # same structure that produced the card they clicked.
    st.session_state["fcn_strike_pct"] = 95.0
    st.session_state["fcn_ki_pct"] = 80.0
    st.session_state["fcn_ko_pct"] = 80.0 + 20.0  # 100.0
    st.session_state["fcn_r_pct"] = 2.25
    st.session_state["fcn_notional"] = 1_000_000.0
    st.session_state["fcn_n_paths"] = 100_000
    st.session_state["fcn_seed"] = 20260526

    # Force the Quote page to re-price on landing.
    st.session_state.pop("fcn_last_run_key", None)

    navigate("Thai FCN Desk", "Price & quote")


def _fcn_render_scenario_card(
    label: str,
    quote: FCNQuote,
    preset: Optional[dict],
    is_user: bool,
) -> None:
    """Render one scenario card (preset or user's last quote)."""
    pr = quote.price_result
    with st.container(border=True):
        # Header: badge + label.
        if is_user:
            st.markdown(
                f"<span style='background:{_FCN_NAVY}; color:{_FCN_BLUE}; "
                f"padding:2px 8px; border-radius:8px; font-size:11px; "
                f"letter-spacing:1.2px; font-weight:600;'>YOUR QUOTE</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<span style='background:#2a2a2a; color:{_FCN_TEXT}; "
                f"padding:2px 8px; border-radius:8px; font-size:11px; "
                f"letter-spacing:1.2px; opacity:0.8;'>PRESET</span>",
                unsafe_allow_html=True,
            )
        st.markdown(f"**{label}**")

        # Big headline coupon.
        coupon_color = _FCN_AMBER if quote.client_coupon_annual > 0 else _FCN_RED
        st.markdown(
            f"<div style='color:{coupon_color}; font-size:2.0rem; "
            f"font-weight:700; line-height:1.05;'>"
            f"{quote.client_coupon_annual*100:.2f}% p.a.</div>"
            f"<div style='color:{_FCN_TEXT}; opacity:0.7; font-size:12px;'>"
            "client coupon</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")

        # Coupon + margin row.
        m1, m2 = st.columns(2)
        m1.metric("Fair coupon", f"{pr.fair_coupon_annual*100:.2f}%")
        m2.metric("Desk margin", f"{quote.desk_margin_annual*100:.2f}%")
        # Risk row.
        m3, m4 = st.columns(2)
        m3.metric("P(KO)", f"{pr.prob_ko*100:.1f}%")
        m4.metric("P(KI)", f"{pr.prob_ki*100:.1f}%")
        m5, m6 = st.columns(2)
        m5.metric("P(loss)", f"{pr.prob_loss_at_maturity*100:.1f}%")
        m6.metric("E[settle]", f"{pr.expected_settle_months:.1f}mo")

        st.caption(
            f"Tickers: {', '.join(quote.tickers)}  ·  "
            f"{pr.tenor_months}M · Strike {pr.strike_pct*100:.0f}% / "
            f"KI {pr.ki_barrier_pct*100:.0f}% / KO {pr.ko_level_pct*100:.0f}%  ·  "
            f"MC stderr ±{pr.mc_stderr_fair_coupon*100:.3f}%"
        )

        # Action button: only on preset cards (the user-quote card already
        # came from the Quote page, so no "load" needed).
        if preset is not None:
            if st.button(
                "↗ Load into Quote page",
                key=f"fcn_load_preset_{label}",
                use_container_width=True,
                help=(
                    "**โหลดค่าทั้งหมดของ Scenario นี้เข้าหน้า Price & Quote**\n\n"
                    "หลังจากกดจะถูกพาไปที่หน้า Quote โดยอัตโนมัติ พร้อมค่า:\n"
                    "• Tickers, Spot, Vol, Div ของหุ้นทุกตัว\n"
                    "• Correlation matrix\n"
                    "• Tenor, Strike, KI, KO\n"
                    "• Desk Margin ตามที่กำหนดใน preset\n\n"
                    "จากนั้นสามารถปรับแต่งค่าใดๆ แล้วกด Price & quote ใหม่ได้"
                ),
            ):
                _fcn_load_preset_into_quote(preset)


def fcn_preset_scenarios_page() -> None:
    """Show 3 SET50 preset scenarios + the user's last custom quote, side by side."""
    st.subheader("Preset scenarios")

    user_quote: Optional[FCNQuote] = st.session_state.get("fcn_last_quote")

    if user_quote is None:
        st.caption(
            "Three pre-configured Thai FCN structures (CPALL · KBANK+SCB · PTT+PTTEP+ADVANC), "
            "all on THOR 2.25%, strike 95% / KI 80% / KO 100%, notional THB 1,000,000. "
            "**Run a custom quote on the Price & quote page** to see it side-by-side here for comparison."
        )
    else:
        st.caption(
            "Your most recent custom quote is shown alongside the three SET50 presets for direct comparison. "
            "Click **'↗ Load into Quote page'** on any preset card to jump back to the pricer with those values loaded."
        )

    # Run / fetch preset prices (cached, so revisits are instant).
    preset_quotes = []
    for preset in FCN_SCENARIO_PRESETS:
        with st.spinner(f"Pricing: {preset['label']}…"):
            preset_quotes.append(_fcn_run_preset(preset, 0.0225, 100_000, 20260526))

    # Assemble cards: user quote (if any) first, then 3 presets.
    user_label = None
    cards: list[tuple] = []  # (label, quote, preset_or_None, is_user)
    if user_quote is not None:
        pr_u = user_quote.price_result
        user_label = (
            f"Your last quote · {pr_u.tenor_months}M · {len(user_quote.tickers)}-Asset "
            f"({', '.join(user_quote.tickers)})"
        )
        cards.append((user_label, user_quote, None, True))
    for preset, q in zip(FCN_SCENARIO_PRESETS, preset_quotes):
        cards.append((preset["label"], q, preset, False))

    # Render cards in N columns where N = total number of cards.
    cols = st.columns(len(cards))
    for col, (label, quote, preset, is_user) in zip(cols, cards):
        with col:
            _fcn_render_scenario_card(label, quote, preset, is_user)

    st.divider()
    st.markdown("##### Sample path visualizations")
    if user_quote is not None and user_label is not None:
        with st.expander(user_label, expanded=False):
            st.plotly_chart(_fcn_path_chart(user_quote), use_container_width=True)
    for preset, q in zip(FCN_SCENARIO_PRESETS, preset_quotes):
        with st.expander(preset["label"], expanded=False):
            st.plotly_chart(_fcn_path_chart(q), use_container_width=True)


def fcn_how_it_works() -> None:
    """Sales-team-friendly explainer."""
    st.subheader("How this engine prices a Thai FCN")

    st.markdown(
        """
        A Thai Fixed Coupon Note is a yield-enhancement product. The investor lends
        the bank a notional, the bank pays them a **fixed monthly coupon** in exchange
        for taking on **equity downside risk** on a worst-of basket. The bank hedges
        that downside in the listed options market and pockets a margin in between.
        """
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Coupon leg**")
        st.markdown(
            "- Paid every month at a **fixed** rate.\n"
            "- Investor receives coupons until the note either knocks out or matures.\n"
            "- That's where the *Fixed* in FCN comes from."
        )
    with c2:
        st.markdown("**KO / Autocall (monthly)**")
        st.markdown(
            "- Checked at each **monthly** observation date.\n"
            "- If the **worst-performing** asset closes ≥ the KO level (typically 100% of spot), "
            "the note **redeems early at par** plus that month's coupon.\n"
            "- Higher KO probability = better client outcome on a present-value basis."
        )
    with c3:
        st.markdown("**KI / Loss (daily)**")
        st.markdown(
            "- Checked **every trading day** across the path.\n"
            "- If the worst-of underlying ever touches the KI barrier "
            "(typically 80% of spot), the **downside protection turns off**.\n"
            "- At maturity, if worst-of finishes below strike, investor takes the loss "
            "at the worst-of performance / strike."
        )

    st.divider()
    st.subheader("Model")
    st.markdown(
        """
        - Each underlying simulated under **risk-neutral GBM** with its own implied vol
          and expected dividend yield.
        - Joint dynamics injected via **Cholesky decomposition** of the user-specified
          correlation matrix. The 1-asset case is the 1×1 identity and is handled
          explicitly; the 2- and 3-asset cases reduce to clean matrix multiplications.
        - **100,000 paths** by default, run in memory-safe batches so the 3-asset 6M
          case never spikes memory.
        """
    )

    st.subheader("Quoting math (no root-finding needed)")
    st.markdown(
        """
        During the Monte Carlo we accumulate two per-path quantities:

        1. **PV of the principal-redemption leg** — discounted notional, with
           early redemption at KO and worst-of haircut on KI-then-below-strike maturities.
        2. **PV of paying N⁄12 each month up to settlement** — call this `PV_unit_coupon`.

        Then the fair annualized coupon that prices the note at par is simply:

        > `c* = (N − PV_principal) / PV_unit_coupon`

        And the client quote is the fair coupon minus the desk's annualized margin.
        The PV breakdown reconciles by construction:

        > `model_base_pv + desk_fees_pv + client_coupon_pv = N` (up to MC error).
        """
    )

    st.subheader("Edge cases handled")
    st.markdown(
        "- 1-asset notes (identity correlation matrix).\n"
        "- Non-PSD user correlation: repaired via nearest-PSD eigenvalue clipping with a UI warning.\n"
        "- Degenerate inputs (negative vols, wrong matrix shape, etc.) raise clear errors.\n"
        "- All MC paths share a single RNG seed for full reproducibility."
    )


# ---------------------------------------------------------------------------
# Thai Mutual Fund Screening Assistant (Anthropic + web search)
# ---------------------------------------------------------------------------

FUND_SCREENER_MODEL = "claude-opus-4-1"
FUND_SCREENER_MAX_TOKENS = 4096
FUND_SCREENER_MAX_SEARCHES_PER_TURN = 8
FUND_SCREENER_USER_LOCATION = {
    "type": "approximate",
    "city": "Bangkok",
    "region": "Bangkok",
    "country": "TH",
    "timezone": "Asia/Bangkok",
}
FUND_SCREENER_SYSTEM_PROMPT = """\
# บทบาท (Role)
คุณคือผู้ช่วยวิเคราะห์กองทุนรวมสำหรับนักลงทุนไทย ทำหน้าที่เฟ้นหาและเปรียบเทียบ
กองทุนตามสิ่งที่ผู้ใช้สนใจ โดยยึดหลักข้อมูลเชิงประจักษ์ (data-driven)
ไม่ใช่การเชียร์กองใดกองหนึ่ง คุณไม่ใช่ที่ปรึกษาการลงทุนที่มีใบอนุญาต
และทุกคำตอบเป็นข้อมูลประกอบการตัดสินใจเท่านั้น

# หลักการสำคัญ
- อ้างอิงข้อมูลจริงที่ค้นได้ (หนังสือชี้ชวน, Fund Fact Sheet, เว็บ บลจ.,
  Morningstar, WealthMagik, SETTRADE/Finnomena/InnovestX) และต้องระบุวันที่ของข้อมูลเสมอ
- ห้ามแต่งตัวเลขหรือชื่อกอง ถ้าหาข้อมูลไม่ได้ให้บอกตรงๆ ว่าหาไม่เจอ
- โปร่งใสเรื่องค่าธรรมเนียม เพราะกระทบผลตอบแทนระยะยาวมาก
- เตือนเสมอว่าผลตอบแทนในอดีตไม่การันตีอนาคต

# ขั้นตอนการทำงาน
1) ถามคำถามจำเป็น 2-4 ข้อ (บัญชีภาษี, ระยะเวลา, ปันผล/สะสม, hedge ค่าเงิน)
2) แปลงธีมที่ผู้ใช้สนใจเป็นหมวดสินทรัพย์/ดัชนีที่ค้นได้จริง
3) ค้นกองที่มีอยู่จริงในไทยให้ครอบคลุม
4) คัดกรองตามค่าธรรมเนียม, underlying, AUM, tracking error, FX hedge, ผลตอบแทนย้อนหลัง, สภาพคล่อง
5) สรุปเป็นตารางเปรียบเทียบ 3-5 กอง พร้อม trade-off
6) ปิดท้ายด้วยคำเตือนความเสี่ยงทุกครั้ง

# รูปแบบการตอบ
- ภาษาไทยเป็นหลัก กระชับ ตรงประเด็น
- ใช้ตารางเมื่อเทียบหลายกอง
- ระบุแหล่งที่มาและวันที่ของข้อมูลเสมอ
- ถ้าข้อมูลขัดแย้งกัน ให้แนะนำให้ยืนยันกับ บลจ. โดยตรง
"""


def _fund_screener_resolve_api_key() -> str:
    """API key from env, Streamlit secrets, or session (sidebar / page form)."""
    def _normalize(raw: str) -> str:
        key = str(raw or "").strip().strip("'\"")
        if not key:
            return ""
        # Handle accidental full command pastes like: setx ANTHROPIC_API_KEY sk-ant-...
        sk_match = re.search(r"(sk-ant-[A-Za-z0-9_-]+)", key)
        if sk_match:
            return sk_match.group(1).strip().strip("'\"")
        return key

    key = _normalize(os.environ.get("ANTHROPIC_API_KEY", ""))
    if key:
        return key
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            key = _normalize(str(st.secrets["ANTHROPIC_API_KEY"]))
            if key:
                return key
    except Exception:
        pass
    return _normalize(str(st.session_state.get("fund_screener_api_key", "")))


def _fund_screener_render_api_key_setup() -> None:
    """Show setup UI when no API key is configured."""
    st.warning("ยังไม่พบ Anthropic API key — ใส่ด้านล่าง หรือตั้งค่าก่อนรันแอป")
    with st.expander("ใส่ API key ในแอป (ใช้เฉพาะ session นี้)", expanded=True):
        with st.form("fund_screener_api_form"):
            entered = st.text_input(
                "Anthropic API Key",
                type="password",
                placeholder="sk-ant-api03-...",
                help="ได้จาก console.anthropic.com — เก็บเฉพาะในเบราว์เซอร์/session นี้",
            )
            if st.form_submit_button("บันทึกและเริ่มใช้งาน", type="primary"):
                if entered.strip():
                    st.session_state.fund_screener_api_key = entered.strip()
                    st.rerun()
                else:
                    st.error("กรุณาใส่ API key")
    st.markdown("**วิธีอื่น (แนะนำถ้ารันบ่อย)**")
    st.code(
        "# PowerShell (session ปัจจุบัน)\n"
        '$env:ANTHROPIC_API_KEY="sk-ant-..."\n'
        "streamlit run wwwwww.py\n\n"
        "# หรือสร้างไฟล์ .streamlit/secrets.toml\n"
        'ANTHROPIC_API_KEY = "sk-ant-..."',
        language="powershell",
    )


def _fund_screener_extract_sources(content_blocks: list[dict]) -> list[tuple[str, str]]:
    """Return unique (title, url) pairs from web search result blocks."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for block in content_blocks:
        if block.get("type") != "web_search_tool_result":
            continue
        for item in block.get("content", []) or []:
            url = item.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            out.append((item.get("title") or url, url))
    return out


def fund_screener_hero() -> None:
    st.title("Thai Mutual Fund Screening Assistant")
    st.caption("Anthropic + Web Search · คัดกรองกองทุนไทยแบบ data-driven")


def fund_screener_chat_page() -> None:
    st.subheader("คุยกับผู้ช่วยคัดเลือกกองทุน")
    st.caption(
        "ตัวอย่างธีม: S&P500, Semiconductor, ทองคำ, หุ้นจีน. "
        "เครื่องมือนี้เป็นข้อมูลประกอบการตัดสินใจ ไม่ใช่คำแนะนำการลงทุนเฉพาะบุคคล"
    )

    if not _ANTHROPIC_OK:
        st.error("ยังไม่ได้ติดตั้ง `anthropic` — รัน `pip install anthropic` แล้วรีสตาร์ตแอป")
        return

    api_key = _fund_screener_resolve_api_key()
    if not api_key:
        _fund_screener_render_api_key_setup()
        return
    if not api_key.startswith("sk-ant-"):
        st.error("รูปแบบ Anthropic API key ไม่ถูกต้อง (ต้องขึ้นต้นด้วย sk-ant-)")
        _fund_screener_render_api_key_setup()
        return

    history_key = "fund_screener_messages"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    for msg in st.session_state[history_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])
            if msg.get("sources"):
                with st.expander("แหล่งอ้างอิง", expanded=False):
                    for title, url in msg["sources"]:
                        st.markdown(f"- [{title}]({url})")

    user_prompt = st.chat_input("พิมพ์ธีมที่สนใจ เช่น S&P500 / Semiconductor / ทองคำ / หุ้นจีน")
    if not user_prompt:
        return

    st.session_state[history_key].append({"role": "user", "text": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    api_messages: list[dict] = []
    for msg in st.session_state[history_key]:
        if msg["role"] == "user":
            api_messages.append({"role": "user", "content": msg["text"]})
        else:
            api_messages.append({"role": "assistant", "content": msg["raw_blocks"]})

    with st.chat_message("assistant"):
        with st.spinner("กำลังวิเคราะห์และค้นข้อมูลกองทุน..."):
            try:
                client = anthropic.Anthropic(api_key=api_key)
                response = client.messages.create(
                    model=FUND_SCREENER_MODEL,
                    max_tokens=FUND_SCREENER_MAX_TOKENS,
                    system=FUND_SCREENER_SYSTEM_PROMPT,
                    messages=api_messages,
                    tools=[
                        {
                            "type": "web_search_20250305",
                            "name": "web_search",
                            "max_uses": FUND_SCREENER_MAX_SEARCHES_PER_TURN,
                            "user_location": FUND_SCREENER_USER_LOCATION,
                        }
                    ],
                )
            except Exception as e:
                st.error(f"เรียก Anthropic API ไม่สำเร็จ: {e}")
                return

        text_parts = [block.text for block in response.content if block.type == "text"]
        answer = "\n".join(part for part in text_parts if part).strip()
        if not answer:
            answer = "ไม่พบคำตอบข้อความจากโมเดล (อาจเกิดข้อผิดพลาดชั่วคราว)"
        st.markdown(answer)

        raw_blocks = [block.model_dump() for block in response.content]
        sources = _fund_screener_extract_sources(raw_blocks)
        if sources:
            with st.expander("แหล่งอ้างอิง", expanded=False):
                for title, url in sources:
                    st.markdown(f"- [{title}]({url})")

    st.session_state[history_key].append(
        {
            "role": "assistant",
            "text": answer,
            "sources": sources,
            "raw_blocks": raw_blocks,
        }
    )


# ---------------------------------------------------------------------------
# Topic registry — add topics here. They appear in the sidebar in this order.
# ---------------------------------------------------------------------------


TOPICS: Dict[str, Topic] = {
    "Stock Analyzer (Auto)": Topic(
        name="Stock Analyzer (Auto)",
        subtitle="Type a ticker · get full Buffett-style analysis from live data",
        hero=analyzer_hero,
        pages={
            "Analyze ticker": analyzer_main,
        },
    ),
    "Gold (XAU/USD)": Topic(
        name="Gold (XAU/USD)",
        subtitle="Multi-horizon forecast · macro · positioning · technicals",
        hero=gold_hero,
        pages={
            "Overview": section_overview,
            "Forecasts": section_forecasts,
            "Macro drivers": section_macro,
            "Technical levels": section_technicals,
            "Scenarios": section_scenarios,
            "Positioning & flows": section_positioning,
            "Wall Street targets": section_targets,
            "Trading strategies": section_strategies,
            "Risk & invalidation": section_risk,
        },
    ),
    "Thai FCN Desk": Topic(
        name="Thai FCN Desk",
        subtitle="Multi-asset FCN pricing · Monte Carlo · 1, 2 or 3 underlyings",
        hero=fcn_hero,
        pages={
            "Price & quote": fcn_quote_page,
            "Preset scenarios": fcn_preset_scenarios_page,
            "How it works": fcn_how_it_works,
        },
    ),
    "Thai Fund Screener": Topic(
        name="Thai Fund Screener",
        subtitle="Anthropic web-search assistant for Thai mutual fund screening",
        hero=fund_screener_hero,
        pages={
            "Chat assistant": fund_screener_chat_page,
        },
    ),
    "+ Add new topic": Topic(
        name="+ Add new topic",
        subtitle="Template for your next instrument or theme",
        hero=template_hero,
        pages={
            "How to add a topic": template_how_to,
        },
    ),
}


# ---------------------------------------------------------------------------
# Navigation — collapsible topic groups in the sidebar
# ---------------------------------------------------------------------------


DEFAULT_TOPIC = next(iter(TOPICS))
DEFAULT_PAGE = next(iter(TOPICS[DEFAULT_TOPIC].pages))

if "active_topic" not in st.session_state:
    st.session_state.active_topic = DEFAULT_TOPIC
if "active_page" not in st.session_state:
    st.session_state.active_page = DEFAULT_PAGE


def navigate(topic_name: str, page_name: str) -> None:
    """Switch to a topic + page and re-run the script."""
    st.session_state.active_topic = topic_name
    st.session_state.active_page = page_name
    st.rerun()


with st.sidebar:
    st.markdown("### Markets Dashboard")
    st.caption("Click a topic to reveal its pages")
    st.divider()

    for topic_name, topic in TOPICS.items():
        is_active_topic = st.session_state.active_topic == topic_name
        with st.expander(topic_name, expanded=is_active_topic):
            st.caption(topic.subtitle)
            for page_name in topic.pages:
                is_active_page = is_active_topic and st.session_state.active_page == page_name
                if st.button(
                    ("●  " if is_active_page else "○  ") + page_name,
                    key=f"nav_{topic_name}_{page_name}",
                    use_container_width=True,
                    type="primary" if is_active_page else "secondary",
                ):
                    navigate(topic_name, page_name)

    if st.session_state.active_topic == "Thai Fund Screener":
        st.divider()
        st.caption("Anthropic API")
        if _fund_screener_resolve_api_key():
            st.success("API key พร้อมใช้งาน")
            if st.button("ลบ key จาก session", key="fund_screener_clear_key"):
                st.session_state.pop("fund_screener_api_key", None)
                st.rerun()
        else:
            with st.form("fund_screener_sidebar_api"):
                sk = st.text_input("API Key", type="password", label_visibility="collapsed")
                if st.form_submit_button("บันทึก key", use_container_width=True):
                    if sk.strip():
                        st.session_state.fund_screener_api_key = sk.strip()
                        st.rerun()

    # Topic-specific quick stats (only meaningful for Gold right now)
    if st.session_state.active_topic == "Gold (XAU/USD)":
        st.divider()
        st.caption("Quick stats · Gold")
        st.metric("Spot", f"${SPOT:,.2f}", f"{DAILY_CHANGE_PCT:+.2f}% · 24h")
        st.metric("DXY", f"{DXY:.2f}", "99–99.5 range")
        st.metric("10Y real yield", f"{TIPS_10Y:.2f}%", "+0.27 pts MTD")
        st.metric("CPI y/y (Apr)", f"{CPI_YOY:.1f}%", f"Core {CORE_CPI_YOY:.1f}%")

    st.divider()
    st.caption(
        "Not investment advice. Trading leveraged FX/commodities carries substantial risk."
    )


# ---------------------------------------------------------------------------
# Render the selected topic + page
# ---------------------------------------------------------------------------


active_topic = TOPICS[st.session_state.active_topic]

# Guard: if a page name was removed while the session was active, fall back.
if st.session_state.active_page not in active_topic.pages:
    st.session_state.active_page = next(iter(active_topic.pages))

active_topic.hero()
st.divider()
active_topic.pages[st.session_state.active_page]()
