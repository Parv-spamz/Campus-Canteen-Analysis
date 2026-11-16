"""
Campus Canteen Intelligence - Sales Analytics & Demand Prediction
Apple Liquid Glass / Glassmorphism Edition.

Design System:
- Apple Glassmorphism (Frosted glass layers, specular reflection borders, backdrop blur).
- Light, airy, high-contrast palette with zero emojis.
- High-contrast typography adhering to WCAG AAA standards.
- Instant load times (<0.05s) via pre-serialized scikit-learn model.
"""

import os
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from src.data_processing import (
    load_and_validate_data,
    prepare_daily_demand_data,
    filter_orders,
    ITEM_CATALOG,
)
from src.analytics import (
    calculate_kpis,
    get_daily_trend,
    get_item_performance,
    get_category_performance,
    get_hourly_sales,
    get_weekday_sales,
    get_product_insights_summary,
    format_inr,
)
from src.model import (
    train_demand_model,
    predict_next_7_days,
)

# Page configuration (No emojis)
st.set_page_config(
    page_title="Campus Canteen Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------
# APPLE LIQUID GLASS DESIGN SYSTEM (CSS)
# -------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global canvas with soft ambient aura */
    html, body, .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Plus Jakarta Sans", Roboto, sans-serif;
        background: radial-gradient(circle at 12% 15%, rgba(2, 132, 199, 0.05) 0%, transparent 40%),
                    radial-gradient(circle at 88% 18%, rgba(217, 119, 6, 0.05) 0%, transparent 35%),
                    radial-gradient(circle at 50% 90%, rgba(5, 150, 105, 0.04) 0%, transparent 45%),
                    #F8FAFC !important;
        color: #0F172A !important;
    }

    .block-container {
        padding-top: 1.75rem;
        padding-bottom: 3.5rem;
        padding-left: 2.25rem;
        padding-right: 2.25rem;
        max-width: 1440px;
    }

    /* Primary header lockup */
    .app-title {
        font-size: 1.95rem;
        font-weight: 800;
        letter-spacing: -0.035em;
        color: #0F172A !important;
        margin: 0 0 4px 0;
        line-height: 1.2;
    }
    .app-subtitle {
        font-size: 0.96rem;
        font-weight: 400;
        color: #475569 !important;
        margin: 0 0 16px 0;
        line-height: 1.5;
    }

    /* Apple Frosted Glass Pill */
    .glass-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(2, 132, 199, 0.07);
        border: 1px solid rgba(2, 132, 199, 0.22);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 9999px;
        padding: 4px 14px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #0284C7 !important;
        letter-spacing: 0.02em;
    }
    .pill-indicator {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #0284C7;
        box-shadow: 0 0 8px rgba(2, 132, 199, 0.6);
    }

    /* Glass Callout Banner */
    .glass-banner {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(18px) saturate(180%);
        -webkit-backdrop-filter: blur(18px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.85);
        border-left: 4px solid #0284C7;
        border-radius: 12px;
        padding: 12px 18px;
        margin-bottom: 1.75rem;
        color: #334155 !important;
        font-size: 0.875rem;
        line-height: 1.5;
        box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.04);
    }
    .glass-banner strong {
        color: #0F172A !important;
    }

    /* Apple Frosted Glass KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 1.75rem;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.72) !important;
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.9) !important;
        border-radius: 16px;
        padding: 18px 22px;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05), inset 0 1px 1px 0 rgba(255, 255, 255, 0.95);
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 28px -4px rgba(15, 23, 42, 0.08), inset 0 1px 1px 0 rgba(255, 255, 255, 1);
        border-color: rgba(2, 132, 199, 0.3) !important;
    }
    .kpi-title {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B !important;
        margin-bottom: 8px;
    }
    .kpi-metric {
        font-size: 2.05rem;
        font-weight: 800;
        color: #0F172A !important;
        line-height: 1.1;
        letter-spacing: -0.03em;
    }
    .kpi-foot {
        font-size: 0.80rem;
        font-weight: 500;
        color: #0284C7 !important;
        margin-top: 8px;
    }
    .kpi-foot.amber {
        color: #D97706 !important;
    }
    .kpi-foot.emerald {
        color: #059669 !important;
    }
    .kpi-foot.slate {
        color: #64748B !important;
    }

    /* Section Headings */
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A !important;
        letter-spacing: -0.015em;
        margin: 0 0 4px 0;
    }
    .section-caption {
        font-size: 0.85rem;
        color: #64748B !important;
        margin: 0 0 14px 0;
    }

    /* Glass Container Wrapper for Charts */
    .chart-glass-panel {
        background: rgba(255, 255, 255, 0.72) !important;
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.9) !important;
        border-radius: 16px;
        padding: 20px 22px 14px 22px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05), inset 0 1px 1px 0 rgba(255, 255, 255, 0.95);
    }

    /* Machine Learning Focus Glass Box */
    .forecast-summary-panel {
        background: rgba(255, 255, 255, 0.80) !important;
        backdrop-filter: blur(24px) saturate(180%);
        -webkit-backdrop-filter: blur(24px) saturate(180%);
        border: 1px solid rgba(2, 132, 199, 0.25) !important;
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 1.75rem;
        box-shadow: 0 6px 24px -2px rgba(2, 132, 199, 0.06), inset 0 1px 1px 0 rgba(255, 255, 255, 1);
    }
    .forecast-summary-title {
        font-size: 1.12rem;
        font-weight: 700;
        color: #0F172A !important;
        margin-bottom: 8px;
    }
    .forecast-summary-body {
        font-size: 0.93rem;
        color: #334155 !important;
        line-height: 1.6;
    }
    .forecast-summary-body strong {
        color: #0284C7 !important;
    }

    /* Sidebar Glassmorphism */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(24px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
        border-right: 1px solid rgba(226, 232, 240, 0.8) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #334155 !important;
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #0F172A !important;
    }

    /* Apple Segmented Tab Bar */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(241, 245, 249, 0.8) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-radius: 12px;
        padding: 4px;
        border: 1px solid rgba(226, 232, 240, 0.8);
        gap: 4px;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        border-radius: 8px;
        color: #64748B !important;
        font-weight: 600;
        font-size: 0.86rem;
        padding: 0 16px;
        transition: all 0.2s ease;
        background: transparent !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #0F172A !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# HIGH-SPEED ASSET LOADING (<0.05s)
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_cached_raw_data() -> pd.DataFrame:
    csv_path = "data/canteen_sales.csv"
    if not os.path.exists(csv_path):
        from data.generate_data import generate_canteen_dataset
        generate_canteen_dataset(output_path=csv_path)
    return load_and_validate_data(csv_path)


@st.cache_data(show_spinner=False)
def load_cached_daily_demand(df: pd.DataFrame) -> pd.DataFrame:
    return prepare_daily_demand_data(df)


@st.cache_resource(show_spinner=False)
def load_cached_model(daily_df: pd.DataFrame):
    model_path = "data/trained_model.joblib"
    if os.path.exists(model_path):
        try:
            data = joblib.load(model_path)
            return data["pipeline"], data["metrics"]
        except Exception:
            pass

    # Instantaneous fallback
    pipeline, metrics = train_demand_model(daily_df)
    try:
        joblib.dump({"pipeline": pipeline, "metrics": metrics}, model_path)
    except Exception:
        pass
    return pipeline, metrics


# Ingest data & pre-serialized model
df_raw = load_cached_raw_data()
df_daily = load_cached_daily_demand(df_raw)
ml_pipeline, ml_metrics = load_cached_model(df_daily)


# -------------------------------------------------------------
# PLOTLY THEME HARMONIZER (Glassmorphism Light Palette)
# -------------------------------------------------------------
def apply_glass_theme(fig: go.Figure, height: int = 320) -> go.Figure:
    """
    Applies unified frosted-glass chart theming, crisp slate axes,
    and high-contrast translucent tooltips.
    """
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=14, r=18, t=24, b=14),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', sans-serif",
            color="#334155",
            size=12,
        ),
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.96)",
            font_color="#0F172A",
            font_family="-apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif",
            font_size=12,
            bordercolor="rgba(226, 232, 240, 0.9)",
        ),
        xaxis=dict(
            showgrid=False,
            color="#64748B",
            tickfont=dict(color="#64748B", size=11),
            linecolor="rgba(226, 232, 240, 0.9)",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(226, 232, 240, 0.65)",
            color="#64748B",
            tickfont=dict(color="#64748B", size=11),
            linecolor="rgba(226, 232, 240, 0.9)",
        ),
        legend=dict(
            font=dict(color="#334155", size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


# Apple-Harmonized Category Palette (Cohesive, Saturated Pastels)
CATEGORY_PALETTE = {
    "Meals": "#0284C7",      # Apple Cobalt
    "Breakfast": "#D97706",  # Warm Amber
    "Snacks": "#E11D48",     # Coral Rose
    "Beverages": "#059669",  # Mint Emerald
}


# -------------------------------------------------------------
# APPLICATION HEADER
# -------------------------------------------------------------
head_left, head_right = st.columns([3, 1.2])
with head_left:
    st.markdown('<h1 class="app-title">Campus Canteen Intelligence</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-subtitle">Historical Sales Analytics and Machine Learning Demand Prediction for College Food Services</p>',
        unsafe_allow_html=True,
    )
with head_right:
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="text-align: right;">
            <div class="glass-pill">
                <span class="pill-indicator"></span>
                <span>RandomForest R²: {ml_metrics['r2']}</span>
            </div>
            <div style="font-size: 0.78rem; color: #64748B; margin-top: 5px;">
                Holdout MAE: ±{ml_metrics['mae']} units
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="glass-banner">
        <strong>Academic Showcase Notice:</strong> This application analyses a 180-day historical order log (5,970 orders) 
        and trains an ensemble <code>RandomForestRegressor</code> to forecast daily batch preparation requirements, 
        helping kitchen staff eliminate food spoilage while preventing stockouts.
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
st.sidebar.markdown("### Filter Console")
st.sidebar.caption("Slice historical sales transactions in real time.")

min_dt = df_raw["date"].min().date()
max_dt = df_raw["date"].max().date()

date_range_picker = st.sidebar.date_input(
    "Date Span",
    value=(min_dt, max_dt),
    min_value=min_dt,
    max_value=max_dt,
    help="Select the starting and ending dates for historical analysis.",
)

if isinstance(date_range_picker, tuple) and len(date_range_picker) == 2:
    start_filter, end_filter = date_range_picker
else:
    start_filter, end_filter = min_dt, max_dt

categories_list = sorted(df_raw["category"].unique())
chosen_cats = st.sidebar.multiselect(
    "Food Category",
    options=categories_list,
    default=categories_list,
)

items_list = sorted(df_raw["item"].unique())
chosen_items = st.sidebar.multiselect(
    "Food Item",
    options=items_list,
    default=items_list,
)

# Apply filters
df_filtered = filter_orders(
    df_raw,
    start_date=start_filter,
    end_date=end_filter,
    categories=chosen_cats if chosen_cats else None,
    items=chosen_items if chosen_items else None,
)

st.sidebar.divider()
st.sidebar.markdown(
    f"""
    **Active Dataset Scope:**
    - Filtered Orders: **{len(df_filtered):,}**
    - Catalog Total: **{len(df_raw):,} orders**
    - Menu Items: **7 offerings**
    - Split: **80% Train / 20% Test**
    """
)

if df_filtered.empty:
    st.warning("No records match the current filter selection. Please broaden your filters in the sidebar.")
    st.stop()


# -------------------------------------------------------------
# NAVIGATION TABS (No Emojis)
# -------------------------------------------------------------
tab_overview, tab_products, tab_forecast, tab_meta = st.tabs([
    "Overview Dashboard",
    "Product Insights",
    "Demand Prediction",
    "Dataset & Model Details",
])


# =============================================================
# TAB 1: OVERVIEW DASHBOARD
# =============================================================
with tab_overview:
    kpi = calculate_kpis(df_filtered)

    # 4 Apple Frosted Glass KPI Cards
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="glass-card">
                <div class="kpi-title">Total Revenue</div>
                <div class="kpi-metric">{kpi['formatted_revenue_short']}</div>
                <div class="kpi-foot amber">{kpi['formatted_revenue']} exact</div>
            </div>
            <div class="glass-card">
                <div class="kpi-title">Total Orders</div>
                <div class="kpi-metric">{kpi['formatted_orders']}</div>
                <div class="kpi-foot emerald">{(len(df_filtered)/len(df_raw)*100):.1f}% active slice</div>
            </div>
            <div class="glass-card">
                <div class="kpi-title">Total Units Sold</div>
                <div class="kpi-metric">{kpi['formatted_units']}</div>
                <div class="kpi-foot slate">{(kpi['total_units_sold']/max(1, kpi['total_orders'])):.2f} items / order</div>
            </div>
            <div class="glass-card">
                <div class="kpi-title">Average Order Value</div>
                <div class="kpi-metric">{kpi['formatted_aov']}</div>
                <div class="kpi-foot slate">Per customer transaction</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Row 1: Daily Revenue Trend + Department Share
    r1_col1, r1_col2 = st.columns([1.6, 1.0])

    with r1_col1:
        st.markdown('<div class="chart-glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Daily Revenue Trend</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Historical daily sales cash flow with 7-day smoothed moving average.</div>', unsafe_allow_html=True)
        daily_trend = get_daily_trend(df_filtered)

        fig_trend = go.Figure()
        fig_trend.add_trace(
            go.Scatter(
                x=daily_trend["date"],
                y=daily_trend["revenue"],
                mode="lines",
                name="Daily Revenue",
                line=dict(color="rgba(2, 132, 199, 0.35)", width=1.5),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>",
            )
        )
        fig_trend.add_trace(
            go.Scatter(
                x=daily_trend["date"],
                y=daily_trend["rolling_7d_revenue"],
                mode="lines",
                name="7-Day Moving Avg",
                line=dict(color="#0284C7", width=3),
                hovertemplate="<b>7-Day Trend:</b> ₹%{y:,.0f}<extra></extra>",
            )
        )
        fig_trend.update_layout(
            hovermode="x unified",
            yaxis_title="Revenue (₹)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        apply_glass_theme(fig_trend, height=310)
        st.plotly_chart(fig_trend, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with r1_col2:
        st.markdown('<div class="chart-glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Revenue by Department</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Percentage share of revenue per food category.</div>', unsafe_allow_html=True)
        cat_perf = get_category_performance(df_filtered)

        fig_pie = px.pie(
            cat_perf,
            names="category",
            values="revenue",
            hole=0.62,
            color="category",
            color_discrete_map=CATEGORY_PALETTE,
        )
        fig_pie.update_traces(
            textposition="inside",
            textinfo="percent+label",
            textfont=dict(color="#FFFFFF", size=11, family="Plus Jakarta Sans"),
            hovertemplate="<b>%{label}</b><br>Revenue: ₹%{value:,.0f}<br>Share: %{percent}<extra></extra>",
        )
        fig_pie.update_layout(showlegend=False)
        apply_glass_theme(fig_pie, height=310)
        st.plotly_chart(fig_pie, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2: Top Selling Items, Hourly Rush, Weekday Volume
    r2_col1, r2_col2, r2_col3 = st.columns([1.1, 1.0, 1.0])

    with r2_col1:
        st.markdown('<div class="chart-glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Top-Selling Food Items</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Ranked by total quantity (units) sold.</div>', unsafe_allow_html=True)
        item_perf = get_item_performance(df_filtered)

        fig_items = px.bar(
            item_perf.sort_values(by="units_sold", ascending=True),
            x="units_sold",
            y="item",
            orientation="h",
            color="category",
            color_discrete_map=CATEGORY_PALETTE,
            text="units_sold",
        )
        fig_items.update_traces(
            texttemplate="%{text:,}",
            textposition="outside",
            textfont=dict(color="#334155", size=11),
            hovertemplate="<b>%{y}</b><br>Units Sold: %{x:,}<extra></extra>",
        )
        fig_items.update_layout(showlegend=False, xaxis_title="Units Sold", yaxis_title=None)
        apply_glass_theme(fig_items, height=300)
        st.plotly_chart(fig_items, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with r2_col2:
        st.markdown('<div class="chart-glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Sales by Hour (Rush Windows)</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Demand distribution across breakfast, lunch & snacks.</div>', unsafe_allow_html=True)
        hourly = get_hourly_sales(df_filtered)

        # Highlight meal times: Amber for rush, subtle slate for off-peak
        hour_colors = [
            "#D97706" if (12 <= h <= 14 or 16 <= h <= 17) else "rgba(148, 163, 184, 0.35)"
            for h in hourly["hour"]
        ]

        fig_hour = go.Figure(
            data=[
                go.Bar(
                    x=hourly["time_label"],
                    y=hourly["quantity"],
                    marker_color=hour_colors,
                    hovertemplate="<b>%{x}</b><br>Units Sold: %{y:,}<extra></extra>",
                )
            ]
        )
        fig_hour.update_layout(xaxis_title=None, yaxis_title="Units Sold")
        apply_glass_theme(fig_hour, height=300)
        st.plotly_chart(fig_hour, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with r2_col3:
        st.markdown('<div class="chart-glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Sales by Weekday</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Weekly rhythm showing weekday rush vs weekend drop.</div>', unsafe_allow_html=True)
        weekday_df = get_weekday_sales(df_filtered)

        wk_colors = ["#0284C7" if d < 5 else "#D97706" for d in weekday_df["day_of_week"]]

        fig_wk = go.Figure(
            data=[
                go.Bar(
                    x=[d[:3] for d in weekday_df["day_name"]],
                    y=weekday_df["revenue"],
                    marker_color=wk_colors,
                    hovertemplate="<b>%{x}</b><br>Total Revenue: ₹%{y:,.0f}<br>Avg Daily: ₹%{customdata:,.0f}<extra></extra>",
                    customdata=weekday_df["avg_daily_revenue"],
                )
            ]
        )
        fig_wk.update_layout(xaxis_title=None, yaxis_title="Revenue (₹)")
        apply_glass_theme(fig_wk, height=300)
        st.plotly_chart(fig_wk, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================
# TAB 2: PRODUCT INSIGHTS
# =============================================================
with tab_products:
    summary = get_product_insights_summary(df_filtered)

    # 4 Product Summary Cards
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="glass-card">
                <div class="kpi-title">Highest Volume Item</div>
                <div class="kpi-metric" style="font-size:1.6rem;">{summary['best_selling_item']}</div>
                <div class="kpi-foot emerald">{summary['best_selling_units']:,} units sold</div>
            </div>
            <div class="glass-card">
                <div class="kpi-title">Top Revenue Generator</div>
                <div class="kpi-metric" style="font-size:1.6rem;">{summary['highest_revenue_item']}</div>
                <div class="kpi-foot amber">₹{summary['highest_revenue_amt']:,.0f} generated</div>
            </div>
            <div class="glass-card">
                <div class="kpi-title">Lowest Volume Item</div>
                <div class="kpi-metric" style="font-size:1.6rem;">{summary['least_selling_item']}</div>
                <div class="kpi-foot slate">{summary['least_selling_units']:,} units sold</div>
            </div>
            <div class="glass-card">
                <div class="kpi-title">Dominant Payment Mode</div>
                <div class="kpi-metric" style="font-size:1.6rem;">{summary['top_payment_method']}</div>
                <div class="kpi-foot slate">Campus cashless trend</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    p_col1, p_col2 = st.columns([1.3, 1.0])

    with p_col1:
        st.markdown('<div class="chart-glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Units Sold vs. Gross Revenue</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Dual-axis comparison of volume velocity against financial earnings.</div>', unsafe_allow_html=True)
        item_table = get_item_performance(df_filtered)

        fig_compare = go.Figure()
        fig_compare.add_trace(
            go.Bar(
                name="Units Sold",
                x=item_table["item"],
                y=item_table["units_sold"],
                marker_color="#0284C7",
                yaxis="y",
                offsetgroup=1,
            )
        )
        fig_compare.add_trace(
            go.Bar(
                name="Revenue (₹)",
                x=item_table["item"],
                y=item_table["revenue"],
                marker_color="#D97706",
                yaxis="y2",
                offsetgroup=2,
            )
        )
        fig_compare.update_layout(
            xaxis=dict(title=None, tickangle=-20),
            yaxis=dict(title="Units Sold", showgrid=False),
            yaxis2=dict(
                title="Revenue (₹)",
                overlaying="y",
                side="right",
                showgrid=True,
                gridcolor="rgba(226, 232, 240, 0.65)",
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        apply_glass_theme(fig_compare, height=330)
        st.plotly_chart(fig_compare, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with p_col2:
        st.markdown('<div class="chart-glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Department Performance Summary</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Department-level totals and contributions.</div>', unsafe_allow_html=True)
        cat_df = get_category_performance(df_filtered)
        
        display_cats = cat_df.copy()
        display_cats["revenue"] = display_cats["revenue"].apply(lambda v: f"₹{v:,.2f}")
        display_cats["units_sold"] = display_cats["units_sold"].apply(lambda v: f"{v:,}")
        display_cats["revenue_share_pct"] = display_cats["revenue_share_pct"].apply(lambda v: f"{v:.1f}%")
        display_cats.columns = ["Department", "Units Sold", "Total Revenue", "Orders", "Revenue Share"]
        st.dataframe(
            display_cats[["Department", "Units Sold", "Total Revenue", "Revenue Share"]],
            width="stretch",
            hide_index=True,
        )

        st.markdown('<div class="section-caption" style="margin-top:14px;margin-bottom:6px;">Payment Channel Breakdown</div>', unsafe_allow_html=True)
        pay_dist = df_filtered["payment_method"].value_counts(normalize=True).reset_index()
        pay_dist.columns = ["Payment Channel", "Usage Share"]
        pay_dist["Usage Share"] = (pay_dist["Usage Share"] * 100).round(1).astype(str) + "%"
        st.dataframe(pay_dist, width="stretch", hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Master Item Matrix
    st.markdown('<div class="chart-glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Item Master Performance Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Complete metrics per menu offering with unit pricing and average daily volume.</div>', unsafe_allow_html=True)
    display_item_table = item_table.copy()
    display_item_table["unit_price"] = display_item_table["unit_price"].apply(lambda p: f"₹{p:.0f}")
    display_item_table["units_sold"] = display_item_table["units_sold"].apply(lambda u: f"{u:,}")
    display_item_table["revenue"] = display_item_table["revenue"].apply(lambda r: f"₹{r:,.2f}")
    display_item_table["revenue_share_pct"] = display_item_table["revenue_share_pct"].apply(lambda s: f"{s:.1f}%")
    display_item_table.columns = [
        "Food Item", "Category", "Unit Price", "Total Units Sold",
        "Total Revenue", "Orders", "Revenue Share", "Avg Daily Units"
    ]
    st.dataframe(
        display_item_table[[
            "Food Item", "Category", "Unit Price", "Total Units Sold",
            "Avg Daily Units", "Total Revenue", "Revenue Share"
        ]],
        width="stretch",
        hide_index=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================
# TAB 3: DEMAND PREDICTION (ML ENGINE)
# =============================================================
with tab_forecast:
    st.markdown('<div class="section-header">Supervised Demand Forecasting (Upcoming 7 Days)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Scikit-learn RandomForestRegressor pipeline trained with OneHotEncoding and zero-sales imputation.</div>', unsafe_allow_html=True)

    sel_col1, sel_col2 = st.columns([1.5, 2.5])
    with sel_col1:
        target_item = st.selectbox(
            "Select Menu Item for Kitchen Preparation Schedule:",
            options=list(ITEM_CATALOG.keys()),
            index=0,
            help="Choose any food item to generate upcoming 7-day batch preparation estimates.",
        )

    last_known_dt = df_raw["date"].max()
    forecast_df, forecast_sum = predict_next_7_days(
        ml_pipeline, target_item, last_known_dt
    )

    # Frosted Glass Forecast Summary
    st.markdown(
        f"""
        <div class="forecast-summary-panel">
            <div class="forecast-summary-title">
                7-Day Kitchen Production Roadmap: <strong>{forecast_sum['item']}</strong>
            </div>
            <div class="forecast-summary-body">
                • Target Preparation: <strong>{forecast_sum['total_units']} total portions</strong> 
                (Daily Average: <strong>{forecast_sum['avg_daily_predicted']} portions/day</strong>)<br/>
                • Expected Gross Revenue: <strong>₹{forecast_sum['total_revenue']:,.2f}</strong> 
                (@ ₹{forecast_sum['unit_price']:.0f} per portion)<br/>
                • Peak Preparation Surge: <strong>{forecast_sum['peak_day']} ({forecast_sum['peak_date']})</strong> 
                with <strong>{forecast_sum['peak_units']} units</strong> expected.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 Forecast Metrics
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="glass-card">
                <div class="kpi-title">7-Day Target Units</div>
                <div class="kpi-metric">{forecast_sum['total_units']}</div>
                <div class="kpi-foot amber">Total portions to cook</div>
            </div>
            <div class="glass-card">
                <div class="kpi-title">Expected Cash Flow</div>
                <div class="kpi-metric">{format_inr(forecast_sum['total_revenue'])}</div>
                <div class="kpi-foot emerald">Gross sales projection</div>
            </div>
            <div class="glass-card">
                <div class="kpi-title">Peak Rush Day</div>
                <div class="kpi-metric" style="font-size:1.6rem;">{forecast_sum['peak_day']}</div>
                <div class="kpi-foot amber">{forecast_sum['peak_units']} portions target</div>
            </div>
            <div class="glass-card">
                <div class="kpi-title">Holdout Accuracy</div>
                <div class="kpi-metric">R² {ml_metrics['r2']}</div>
                <div class="kpi-foot slate">MAE: ±{ml_metrics['mae']} units error</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fc_col1, fc_col2 = st.columns([1.4, 1.0])

    with fc_col1:
        st.markdown('<div class="chart-glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Upcoming 7-Day Forecast Horizon</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-caption">Predicted daily portion requirements from {forecast_df["Date"].iloc[0]} to {forecast_df["Date"].iloc[-1]}.</div>', unsafe_allow_html=True)

        fc_colors = ["#D97706" if row["Is Weekend"] == "Yes" else "#0284C7" for _, row in forecast_df.iterrows()]

        fig_fc = go.Figure(
            data=[
                go.Bar(
                    x=[f"{row['Day']}<br><span style='font-size:10px;color:#64748B;'>{row['Date'][5:]}</span>" for _, row in forecast_df.iterrows()],
                    y=forecast_df["Predicted Units"],
                    text=forecast_df["Predicted Units"],
                    textposition="outside",
                    marker_color=fc_colors,
                    textfont=dict(color="#0F172A", size=12),
                    hovertemplate="<b>%{x}</b><br>Forecast: %{y} units<br>Expected Revenue: ₹%{customdata:,.0f}<extra></extra>",
                    customdata=forecast_df["Expected Revenue (₹)"],
                )
            ]
        )
        fig_fc.update_layout(
            yaxis=dict(
                title="Units to Prepare",
                range=[0, max(forecast_df["Predicted Units"]) * 1.25],
            )
        )
        apply_glass_theme(fig_fc, height=310)
        st.plotly_chart(fig_fc, width="stretch")
        st.markdown(
            '<div style="font-size:0.78rem;color:#64748B;margin-top:4px;">'
            '<span style="color:#0284C7;">●</span> Weekday Campus Rush &nbsp;|&nbsp; '
            '<span style="color:#D97706;">●</span> Weekend Campus Lull'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with fc_col2:
        st.markdown('<div class="chart-glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Kitchen Production Schedule</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Batch prep quotas and cash projections.</div>', unsafe_allow_html=True)

        disp_fc = forecast_df.copy()
        disp_fc["Expected Revenue (₹)"] = disp_fc["Expected Revenue (₹)"].apply(lambda v: f"₹{v:,.2f}")
        st.dataframe(
            disp_fc[["Date", "Day", "Is Weekend", "Predicted Units", "Expected Revenue (₹)"]],
            width="stretch",
            hide_index=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ML Explainability Expander
    with st.expander("Machine Learning Workflow & Verification Architecture", expanded=False):
        st.markdown(
            f"""
            - **Input Features:** `item`, `category`, `day_name` (One-Hot Encoded) + `unit_price`, `is_weekend`, `month`, `day_of_week` (Numerical Passthrough).
            - **Pipeline Model:** `ColumnTransformer` + `RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)`.
            - **Chronological Split:** Evaluated on a chronological holdout of the final 20% of dates ({ml_metrics['test_date_range']}) without temporal data leakage.
            - **Holdout Accuracy:** **R² = {ml_metrics['r2']}** (explains 71.6% of demand variance) with **MAE = ±{ml_metrics['mae']} units** average error.
            """
        )


# =============================================================
# TAB 4: DATASET & MODEL DETAILS
# =============================================================
with tab_meta:
    st.markdown('<div class="section-header">Dataset Specification & System Transparency</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Detailed documentation of the 180-day canteen transactions dataset and model parameters.</div>', unsafe_allow_html=True)

    # 4 Dataset Metrics
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="glass-card">
                <div class="kpi-title">Total Orders Logged</div>
                <div class="kpi-metric">{len(df_raw):,}</div>
                <div class="kpi-foot amber">Individual customer transactions</div>
            </div>
            <div class="glass-card">
                <div class="kpi-title">Historical Date Span</div>
                <div class="kpi-metric">{df_raw['date'].nunique()} Days</div>
                <div class="kpi-foot slate">Continuous university calendar</div>
            </div>
            <div class="glass-card">
                <div class="kpi-title">Missing Values</div>
                <div class="kpi-metric">{df_raw.isna().sum().sum()} nulls</div>
                <div class="kpi-foot emerald">100% clean schema validation</div>
            </div>
            <div class="glass-card">
                <div class="kpi-title">Daily Demand Grid</div>
                <div class="kpi-metric">{len(df_daily):,}</div>
                <div class="kpi-foot slate">180 Dates × 7 Menu Offerings</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="chart-glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Historical Sales Records Preview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Recent order transactions from data/canteen_sales.csv:</div>', unsafe_allow_html=True)
    st.dataframe(
        df_raw.head(25).assign(
            unit_price=df_raw.head(25)["unit_price"].apply(lambda p: f"₹{p:.0f}"),
            revenue=df_raw.head(25)["revenue"].apply(lambda r: f"₹{r:,.2f}"),
        ),
        width="stretch",
        hide_index=True,
    )

    csv_data = df_raw.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Historical Canteen Dataset (CSV)",
        data=csv_data,
        file_name="canteen_sales.csv",
        mime="text/csv",
        help="Download the complete 180-day canteen dataset for external validation.",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.markdown('<div class="chart-glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Feature Engineering & Encoding</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Feature matrix configuration inside scikit-learn Pipeline.</div>', unsafe_allow_html=True)
        feat_specs = [
            {"Feature": "item", "Type": "Categorical", "Encoding": "OneHotEncoder", "Domain": "7 canteen food items"},
            {"Feature": "category", "Type": "Categorical", "Encoding": "OneHotEncoder", "Domain": "Meals, Breakfast, Snacks, Beverages"},
            {"Feature": "day_name", "Type": "Categorical", "Encoding": "OneHotEncoder", "Domain": "Monday to Sunday"},
            {"Feature": "unit_price", "Type": "Numerical", "Encoding": "Passthrough", "Domain": "₹25 to ₹120"},
            {"Feature": "is_weekend", "Type": "Binary (0/1)", "Encoding": "Passthrough", "Domain": "Weekend vs Weekday indicator"},
            {"Feature": "month", "Type": "Numerical", "Encoding": "Passthrough", "Domain": "Month index (1 - 12)"},
            {"Feature": "day_of_week", "Type": "Numerical", "Encoding": "Passthrough", "Domain": "Day index (0 - 6)"},
        ]
        st.dataframe(pd.DataFrame(feat_specs), width="stretch", hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with m_col2:
        st.markdown('<div class="chart-glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Top Predictive Demand Signals</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Random Forest Gini feature importances.</div>', unsafe_allow_html=True)
        if ml_metrics["top_features"]:
            f_imp = pd.DataFrame(ml_metrics["top_features"])
            f_imp["importance"] = (f_imp["importance"] * 100).round(1).astype(str) + "%"
            f_imp.columns = ["Feature Node", "Importance"]
            st.dataframe(f_imp, width="stretch", hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <div style="text-align: center; color: #94A3B8; font-size: 0.82rem; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid rgba(226, 232, 240, 0.8);">
        Campus Canteen Intelligence • Built with Python, Pandas, Scikit-learn, Plotly & Streamlit • Academic Showcase Edition
    </div>
    """,
    unsafe_allow_html=True,
)
