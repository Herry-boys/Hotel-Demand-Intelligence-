"""
Hotel Demand Intelligence Dashboard
====================================
A production-ready Streamlit web app for analyzing hotel booking data,
visualizing demand patterns, and predicting cancellations using ML models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hotel Demand Intelligence",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0f1b2d 0%, #1a2f4a 50%, #0d2137 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(100, 180, 255, 0.15);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        font-family: 'Playfair Display', serif;
        font-size: 2.4rem;
        font-weight: 900;
        color: #f0f6ff;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #8bb8e8;
        font-size: 1rem;
        margin: 0.4rem 0 0 0;
        font-weight: 300;
    }
    .header-badge {
        display: inline-block;
        background: rgba(100,180,255,0.15);
        border: 1px solid rgba(100,180,255,0.3);
        color: #64b4ff;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(145deg, #0f1b2d, #162438);
        border: 1px solid rgba(100,180,255,0.12);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-2px); }
    .kpi-label {
        color: #8bb8e8;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        font-weight: 700;
        color: #f0f6ff;
        line-height: 1;
    }
    .kpi-delta {
        font-size: 0.78rem;
        color: #52c87e;
        margin-top: 0.3rem;
        font-weight: 500;
    }
    .kpi-delta.negative { color: #ff6b6b; }

    /* Section Headers */
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #e8f2ff;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(100,180,255,0.2);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1623 0%, #0f1b2d 100%);
        border-right: 1px solid rgba(100,180,255,0.1);
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stDateInput label,
    [data-testid="stSidebar"] .stSlider label {
        color: #8bb8e8 !important;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }

    /* Prediction card */
    .pred-safe {
        background: linear-gradient(135deg, #0d2b1a, #0f3320);
        border: 1px solid #2d7a4f;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .pred-risk {
        background: linear-gradient(135deg, #2b0d0d, #331010);
        border: 1px solid #7a2d2d;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .pred-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #4a6a8a;
        font-size: 0.78rem;
        margin-top: 3rem;
        padding: 1.5rem;
        border-top: 1px solid rgba(100,180,255,0.1);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Metric overrides */
    [data-testid="metric-container"] {
        background: linear-gradient(145deg, #0f1b2d, #162438);
        border: 1px solid rgba(100,180,255,0.12);
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ─────────────────────────────────────────────────────────────
DATA_URL = "https://raw.githubusercontent.com/Herry-boys/fileupload/refs/heads/main/hotel_bookings_updated_2024.csv"

MONTH_ORDER = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]

@st.cache_data(show_spinner=False)
def load_data(url: str) -> pd.DataFrame:
    """Load and return raw hotel bookings dataset with caching."""
    df = pd.read_csv(url)
    return df


@st.cache_data(show_spinner=False)
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and engineer features from raw data."""
    df = df.copy()

    # ── Missing values ──────────────────────────────────────────────────────
    df["children"]  = df["children"].fillna(0)
    df["agent"]     = df["agent"].fillna(0)
    df["company"]   = df["company"].fillna(0)
    df["country"]   = df["country"].fillna("Unknown")

    # Drop rows where ADR is negative (data error)
    df = df[df["adr"] >= 0]

    # ── Date features ───────────────────────────────────────────────────────
    df["arrival_date_month"] = pd.Categorical(
        df["arrival_date_month"], categories=MONTH_ORDER, ordered=True
    )
    # Build proper arrival date
    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str) + "-" +
        df["arrival_date_month"].astype(str) + "-" +
        df["arrival_date_day_of_month"].astype(str),
        format="%Y-%B-%d", errors="coerce"
    )

    # ── Derived features ────────────────────────────────────────────────────
    df["total_nights"]  = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"]  = df["adults"] + df["children"] + df["babies"]
    df["revenue"]       = df["adr"] * df["total_nights"]
    df["is_family"]     = ((df["children"] > 0) | (df["babies"] > 0)).astype(int)

    # Hotel type (simplified label)
    df["hotel_type"] = df["hotel"].apply(
        lambda x: "City Hotel" if "City" in str(x) else "Resort Hotel"
    )

    return df


# ─── ML Model ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    """Train an XGBoost cancellation predictor and return model + encoders."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder

    feature_cols = [
        "lead_time", "total_nights", "total_guests", "adr",
        "booking_changes", "previous_cancellations",
        "total_of_special_requests", "is_repeated_guest",
        "days_in_waiting_list", "required_car_parking_spaces",
        "hotel_type", "market_segment", "deposit_type", "customer_type"
    ]

    cat_cols = ["hotel_type", "market_segment", "deposit_type", "customer_type"]
    encoders = {}
    df_model = df[feature_cols + ["is_canceled"]].dropna()

    for col in cat_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        encoders[col] = le

    X = df_model[feature_cols]
    y = df_model["is_canceled"]

    model = GradientBoostingClassifier(n_estimators=120, max_depth=4,
                                       learning_rate=0.1, random_state=42)
    model.fit(X, y)
    return model, encoders, feature_cols, cat_cols


# ─── Visualization Helpers ────────────────────────────────────────────────────
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,27,45,0.6)",
    font=dict(family="DM Sans", color="#c8dff5"),
    xaxis=dict(gridcolor="rgba(100,180,255,0.08)", linecolor="rgba(100,180,255,0.15)"),
    yaxis=dict(gridcolor="rgba(100,180,255,0.08)", linecolor="rgba(100,180,255,0.15)"),
    margin=dict(l=10, r=10, t=40, b=10),
)
COLOR_SEQ = ["#3b9eff", "#52c87e", "#ffb547", "#ff6b9d", "#a78bfa", "#34d3c7"]


def apply_theme(fig):
    fig.update_layout(**PLOTLY_THEME)
    return fig


def booking_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Monthly bookings trend with cancellation overlay."""
    monthly = (
        df.groupby(["arrival_date_month", "is_canceled"], observed=True)
          .size()
          .reset_index(name="count")
    )
    monthly["status"] = monthly["is_canceled"].map({0: "Confirmed", 1: "Cancelled"})

    fig = px.bar(
        monthly, x="arrival_date_month", y="count", color="status",
        title="Monthly Bookings: Confirmed vs Cancelled",
        labels={"arrival_date_month": "Month", "count": "Bookings"},
        color_discrete_map={"Confirmed": "#3b9eff", "Cancelled": "#ff6b6b"},
        barmode="stack",
    )
    fig.update_layout(**PLOTLY_THEME, title_font_size=15,
                      legend=dict(orientation="h", y=1.05, x=0))
    return fig


def cancellation_by_segment_chart(df: pd.DataFrame) -> go.Figure:
    """Cancellation rate by market segment."""
    seg = (
        df.groupby("market_segment")["is_canceled"]
          .agg(["sum", "count"])
          .reset_index()
    )
    seg["cancel_rate"] = (seg["sum"] / seg["count"] * 100).round(1)
    seg = seg.sort_values("cancel_rate", ascending=True)

    fig = px.bar(
        seg, x="cancel_rate", y="market_segment", orientation="h",
        title="Cancellation Rate by Market Segment (%)",
        labels={"cancel_rate": "Cancellation Rate (%)", "market_segment": ""},
        color="cancel_rate",
        color_continuous_scale=[[0, "#52c87e"], [0.5, "#ffb547"], [1, "#ff6b6b"]],
    )
    fig.update_layout(**PLOTLY_THEME, title_font_size=15,
                      coloraxis_showscale=False)
    return fig


def adr_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Average Daily Rate (ADR) trend by month and hotel type."""
    adr_df = (
        df.groupby(["arrival_date_month", "hotel_type"], observed=True)["adr"]
          .mean()
          .reset_index()
    )
    fig = px.line(
        adr_df, x="arrival_date_month", y="adr", color="hotel_type",
        title="Average Daily Rate (ADR) by Month",
        labels={"arrival_date_month": "Month", "adr": "ADR (₹)", "hotel_type": "Hotel Type"},
        markers=True,
        color_discrete_sequence=["#3b9eff", "#ffb547"],
    )
    fig.update_traces(line_width=2.5)
    fig.update_layout(**PLOTLY_THEME, title_font_size=15,
                      legend=dict(orientation="h", y=1.05, x=0))
    return fig


def lead_time_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Lead time distribution by cancellation status."""
    fig = px.histogram(
        df[df["lead_time"] <= 365], x="lead_time",
        color=df[df["lead_time"] <= 365]["is_canceled"].map({0: "Confirmed", 1: "Cancelled"}),
        nbins=50, barmode="overlay", opacity=0.75,
        title="Lead Time Distribution: Confirmed vs Cancelled",
        labels={"lead_time": "Lead Time (days)", "count": "Bookings"},
        color_discrete_map={"Confirmed": "#3b9eff", "Cancelled": "#ff6b6b"},
    )
    fig.update_layout(**PLOTLY_THEME, title_font_size=15,
                      legend=dict(orientation="h", y=1.05, x=0))
    return fig


def revenue_by_hotel_chart(df: pd.DataFrame) -> go.Figure:
    """Total revenue per hotel type per month."""
    rev = (
        df[df["total_nights"] > 0]
          .groupby(["arrival_date_month", "hotel_type"], observed=True)["revenue"]
          .sum()
          .reset_index()
    )
    fig = px.area(
        rev, x="arrival_date_month", y="revenue", color="hotel_type",
        title="Estimated Monthly Revenue by Hotel Type",
        labels={"arrival_date_month": "Month", "revenue": "Revenue (₹)", "hotel_type": "Hotel Type"},
        color_discrete_sequence=["#3b9eff", "#ffb547"],
    )
    fig.update_layout(**PLOTLY_THEME, title_font_size=15,
                      legend=dict(orientation="h", y=1.05, x=0))
    return fig


def customer_type_pie(df: pd.DataFrame) -> go.Figure:
    """Customer type breakdown donut chart."""
    ct = df["customer_type"].value_counts().reset_index()
    ct.columns = ["type", "count"]
    fig = px.pie(
        ct, values="count", names="type", hole=0.55,
        title="Customer Type Distribution",
        color_discrete_sequence=COLOR_SEQ,
    )
    fig.update_layout(**PLOTLY_THEME, title_font_size=15,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.2))
    return fig


def special_requests_chart(df: pd.DataFrame) -> go.Figure:
    """Cancellation rate by number of special requests."""
    sr = (
        df.groupby("total_of_special_requests")["is_canceled"]
          .agg(["sum", "count"])
          .reset_index()
    )
    sr["cancel_rate"] = (sr["sum"] / sr["count"] * 100).round(1)
    fig = px.bar(
        sr, x="total_of_special_requests", y="cancel_rate",
        title="Cancellation Rate vs. Number of Special Requests",
        labels={"total_of_special_requests": "Special Requests", "cancel_rate": "Cancellation Rate (%)"},
        color="cancel_rate",
        color_continuous_scale=[[0, "#52c87e"], [0.5, "#ffb547"], [1, "#ff6b6b"]],
    )
    fig.update_layout(**PLOTLY_THEME, title_font_size=15, coloraxis_showscale=False)
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar(df: pd.DataFrame):
    """Render sidebar filters and return filtered dataframe."""
    with st.sidebar:
        st.markdown("""
        <div style='padding:1rem 0 0.5rem 0;'>
            <div style='font-family:Playfair Display,serif;font-size:1.3rem;
                        font-weight:700;color:#e8f2ff;'>🏨 Filters</div>
            <div style='color:#4a6a8a;font-size:0.78rem;margin-top:0.2rem;'>
                Refine the dashboard data
            </div>
        </div>
        <hr style='border-color:rgba(100,180,255,0.1);margin:0.5rem 0 1rem 0;'>
        """, unsafe_allow_html=True)

        # Hotel Type
        hotel_types = ["All"] + sorted(df["hotel_type"].dropna().unique().tolist())
        hotel_filter = st.selectbox("Hotel Type", hotel_types)

        # Market Segment
        segments = sorted(df["market_segment"].dropna().unique().tolist())
        seg_filter = st.multiselect(
            "Market Segment",
            options=segments,
            default=segments,
            help="Select one or more market segments"
        )

        # Room Type
        room_types = sorted(df["reserved_room_type"].dropna().unique().tolist())
        room_filter = st.multiselect(
            "Reserved Room Type",
            options=room_types,
            default=room_types,
        )

        # Lead Time Range
        st.markdown("<br>", unsafe_allow_html=True)
        lead_min = int(df["lead_time"].min())
        lead_max = int(df["lead_time"].max())
        lead_range = st.slider(
            "Lead Time (days)",
            min_value=lead_min,
            max_value=min(lead_max, 500),
            value=(lead_min, min(lead_max, 365)),
        )

        # Cancellation filter
        cancel_opt = st.radio(
            "Booking Status",
            options=["All", "Confirmed Only", "Cancelled Only"],
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("📊 Data source: Hotel Booking Demand Dataset")

    # ── Apply Filters ──────────────────────────────────────────────────────
    fdf = df.copy()

    if hotel_filter != "All":
        fdf = fdf[fdf["hotel_type"] == hotel_filter]

    if seg_filter:
        fdf = fdf[fdf["market_segment"].isin(seg_filter)]

    if room_filter:
        fdf = fdf[fdf["reserved_room_type"].isin(room_filter)]

    fdf = fdf[(fdf["lead_time"] >= lead_range[0]) & (fdf["lead_time"] <= lead_range[1])]

    if cancel_opt == "Confirmed Only":
        fdf = fdf[fdf["is_canceled"] == 0]
    elif cancel_opt == "Cancelled Only":
        fdf = fdf[fdf["is_canceled"] == 1]

    return fdf


# ─── KPI Section ──────────────────────────────────────────────────────────────
def render_kpis(df: pd.DataFrame):
    """Render KPI cards row."""
    total = len(df)
    cancel_rate = df["is_canceled"].mean() * 100 if total > 0 else 0
    avg_adr = df["adr"].mean() if total > 0 else 0
    avg_nights = df["total_nights"].mean() if total > 0 else 0
    total_rev = df["revenue"].sum() if total > 0 else 0
    repeat_rate = df["is_repeated_guest"].mean() * 100 if total > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("Total Bookings", f"{total:,}")
    with c2:
        st.metric("Cancellation Rate", f"{cancel_rate:.1f}%",
                  delta=f"{cancel_rate - 37:.1f}% vs avg",
                  delta_color="inverse")
    with c3:
        st.metric("Avg Daily Rate", f"₹{avg_adr:,.0f}")
    with c4:
        st.metric("Avg Stay Length", f"{avg_nights:.1f} nights")
    with c5:
        st.metric("Repeat Guests", f"{repeat_rate:.1f}%",
                  delta="↑ loyalty" if repeat_rate > 3 else "↓ loyalty")


# ─── Prediction Section ───────────────────────────────────────────────────────
def render_prediction_tab(df: pd.DataFrame):
    """Render the cancellation prediction form and output."""
    st.markdown('<div class="section-title">🤖 Cancellation Risk Predictor</div>',
                unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8bb8e8;font-size:0.9rem;margin-bottom:1.5rem;'>"
        "Fill in booking details to get an instant cancellation risk prediction "
        "powered by Gradient Boosting.</p>",
        unsafe_allow_html=True
    )

    with st.spinner("⚙️ Loading prediction model…"):
        model, encoders, feature_cols, cat_cols = train_model(df)

    col_a, col_b = st.columns(2)
    with col_a:
        lead_time        = st.slider("Lead Time (days)", 0, 365, 60)
        total_nights     = st.slider("Total Stay Nights", 0, 20, 3)
        total_guests     = st.slider("Total Guests", 1, 10, 2)
        adr              = st.number_input("Average Daily Rate (₹)", 0.0, 5000.0, 120.0, 10.0)
        booking_changes  = st.slider("Booking Changes", 0, 10, 0)
        prev_cancel      = st.slider("Previous Cancellations", 0, 10, 0)

    with col_b:
        special_requests = st.slider("Total Special Requests", 0, 5, 1)
        is_repeated      = st.selectbox("Repeated Guest?", [0, 1],
                                        format_func=lambda x: "Yes" if x else "No")
        waiting_list     = st.slider("Days in Waiting List", 0, 200, 0)
        parking_spaces   = st.slider("Required Parking Spaces", 0, 3, 0)
        hotel_type_pred  = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
        market_seg_pred  = st.selectbox("Market Segment",
                                        sorted(df["market_segment"].dropna().unique()))
        deposit_pred     = st.selectbox("Deposit Type",
                                        sorted(df["deposit_type"].dropna().unique()))
        cust_type_pred   = st.selectbox("Customer Type",
                                        sorted(df["customer_type"].dropna().unique()))

    if st.button("🔮 Predict Cancellation Risk", use_container_width=True):
        input_data = {
            "lead_time": lead_time,
            "total_nights": total_nights,
            "total_guests": total_guests,
            "adr": adr,
            "booking_changes": booking_changes,
            "previous_cancellations": prev_cancel,
            "total_of_special_requests": special_requests,
            "is_repeated_guest": is_repeated,
            "days_in_waiting_list": waiting_list,
            "required_car_parking_spaces": parking_spaces,
            "hotel_type": hotel_type_pred,
            "market_segment": market_seg_pred,
            "deposit_type": deposit_pred,
            "customer_type": cust_type_pred,
        }
        row = pd.DataFrame([input_data])

        # Encode categoricals
        for col in cat_cols:
            le = encoders[col]
            val = row[col].astype(str)
            known = set(le.classes_)
            row[col] = val.apply(lambda x: le.transform([x])[0] if x in known else 0)

        pred = model.predict(row[feature_cols])[0]
        proba = model.predict_proba(row[feature_cols])[0]

        p_cancel = proba[1] * 100
        p_keep   = proba[0] * 100

        st.markdown("<br>", unsafe_allow_html=True)
        if pred == 1:
            st.markdown(f"""
            <div class="pred-risk">
                <div style='font-size:2rem;'>⚠️</div>
                <div class="pred-title" style='color:#ff6b6b;'>HIGH CANCELLATION RISK</div>
                <div style='color:#ffaaaa;font-size:0.95rem;margin-top:0.5rem;'>
                    This booking is likely to be <strong>cancelled</strong>
                </div>
                <div style='margin-top:1rem;display:flex;justify-content:center;gap:2rem;'>
                    <div>
                        <div style='color:#ff6b6b;font-size:1.8rem;font-weight:700;'>{p_cancel:.1f}%</div>
                        <div style='color:#8b9aaa;font-size:0.75rem;'>CANCEL PROBABILITY</div>
                    </div>
                    <div>
                        <div style='color:#52c87e;font-size:1.8rem;font-weight:700;'>{p_keep:.1f}%</div>
                        <div style='color:#8b9aaa;font-size:0.75rem;'>KEEP PROBABILITY</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="pred-safe">
                <div style='font-size:2rem;'>✅</div>
                <div class="pred-title" style='color:#52c87e;'>LOW CANCELLATION RISK</div>
                <div style='color:#aaffcc;font-size:0.95rem;margin-top:0.5rem;'>
                    This booking is likely to be <strong>confirmed</strong>
                </div>
                <div style='margin-top:1rem;display:flex;justify-content:center;gap:2rem;'>
                    <div>
                        <div style='color:#52c87e;font-size:1.8rem;font-weight:700;'>{p_keep:.1f}%</div>
                        <div style='color:#8b9aaa;font-size:0.75rem;'>KEEP PROBABILITY</div>
                    </div>
                    <div>
                        <div style='color:#ff6b6b;font-size:1.8rem;font-weight:700;'>{p_cancel:.1f}%</div>
                        <div style='color:#8b9aaa;font-size:0.75rem;'>CANCEL PROBABILITY</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Risk gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p_cancel,
            title={"text": "Cancellation Risk Score", "font": {"size": 14, "color": "#8bb8e8"}},
            number={"suffix": "%", "font": {"size": 28, "color": "#f0f6ff"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#4a6a8a"},
                "bar": {"color": "#ff6b6b" if pred == 1 else "#52c87e"},
                "steps": [
                    {"range": [0, 33],  "color": "rgba(82,200,126,0.15)"},
                    {"range": [33, 66], "color": "rgba(255,181,71,0.15)"},
                    {"range": [66, 100],"color": "rgba(255,107,107,0.15)"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "value": p_cancel},
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#c8dff5"),
                                height=220, margin=dict(l=20,r=20,t=40,b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)


# ─── Insights Section ─────────────────────────────────────────────────────────
def render_insights():
    """Render static business insights."""
    st.markdown('<div class="section-title">💡 Key Business Insights</div>',
                unsafe_allow_html=True)
    insights = [
        ("📅", "High Lead Time = High Cancellation Risk",
         "Bookings made far in advance have a significantly higher cancellation rate. "
         "Consider implementing flexible policies and overbooking strategies for early reservations."),
        ("🌐", "Online Segment Has Highest Cancellation Rate",
         "Online TA bookings cancel more than any other segment. "
         "Offer non-refundable discount options to reduce last-minute cancellations."),
        ("🌞", "Peak Season: July & August",
         "These months see the highest booking volumes. Apply dynamic pricing during peak "
         "months to maximize revenue."),
        ("⭐", "Special Requests Reduce Cancellations",
         "Guests who make special requests are far less likely to cancel. Encourage guests "
         "to add requests at booking time to increase commitment."),
        ("🔄", "Repeat Guests Rarely Cancel",
         "Repeat guests have a cancellation rate close to zero. Invest in loyalty programs "
         "to convert one-time customers into repeat guests."),
        ("🏔️", "Resort Hotels Have Higher Off-Season Cancellations",
         "Offer winter discounts or holiday packages to boost off-season bookings "
         "and reduce cancellation rates for resort properties."),
    ]
    cols = st.columns(2)
    for i, (icon, title, desc) in enumerate(insights):
        with cols[i % 2]:
            st.markdown(f"""
            <div style='background:linear-gradient(145deg,#0f1b2d,#162438);
                        border:1px solid rgba(100,180,255,0.12);border-radius:10px;
                        padding:1.1rem 1.2rem;margin-bottom:1rem;'>
                <div style='font-size:1.5rem;margin-bottom:0.4rem;'>{icon}</div>
                <div style='color:#e8f2ff;font-weight:600;font-size:0.9rem;
                            margin-bottom:0.4rem;'>{title}</div>
                <div style='color:#6a8aaa;font-size:0.82rem;line-height:1.5;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ─── Main App ─────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <div class="header-badge">✦ AI-Powered Analytics</div>
        <h1>Hotel Demand Intelligence Dashboard</h1>
        <p>Real-time insights into bookings, cancellations, revenue & demand patterns</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    with st.spinner("⏳ Loading hotel booking data…"):
        try:
            raw_df = load_data(DATA_URL)
            df = preprocess_data(raw_df)
        except Exception as e:
            st.error(f"❌ Failed to load data: {e}")
            st.info("Please check your internet connection and try again.")
            st.stop()

    # Sidebar filters → filtered dataframe
    fdf = render_sidebar(df)

    if fdf.empty:
        st.warning("⚠️ No data matches your current filter selection. Please adjust the filters.")
        st.stop()

    # ── KPIs ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Key Performance Indicators</div>',
                unsafe_allow_html=True)
    render_kpis(fdf)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main Tabs ─────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Booking Trends",
        "❌ Cancellation Analysis",
        "💰 Revenue & ADR",
        "🤖 Predict Cancellation",
        "💡 Insights",
    ])

    # ── Tab 1: Booking Trends ─────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-title">Booking Trends Over Time</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(booking_trend_chart(fdf), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(lead_time_distribution_chart(fdf), use_container_width=True)
        with c2:
            st.plotly_chart(customer_type_pie(fdf), use_container_width=True)

        # Dataset preview
        if st.checkbox("👁️ Show Raw Dataset Preview"):
            st.dataframe(
                fdf.head(200).style.format({"adr": "₹{:.2f}", "revenue": "₹{:.2f}"}),
                use_container_width=True,
                height=300
            )

    # ── Tab 2: Cancellation Analysis ──────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-title">Cancellation Rate Analysis</div>',
                    unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(cancellation_by_segment_chart(fdf), use_container_width=True)
        with c2:
            # Cancellation by deposit type
            dep = (
                fdf.groupby("deposit_type")["is_canceled"]
                   .agg(["sum","count"])
                   .reset_index()
            )
            dep["cancel_rate"] = (dep["sum"] / dep["count"] * 100).round(1)
            fig_dep = px.bar(
                dep, x="deposit_type", y="cancel_rate",
                title="Cancellation Rate by Deposit Type (%)",
                labels={"deposit_type": "Deposit Type", "cancel_rate": "Rate (%)"},
                color="cancel_rate",
                color_continuous_scale=[[0,"#52c87e"],[0.5,"#ffb547"],[1,"#ff6b6b"]],
            )
            fig_dep.update_layout(**PLOTLY_THEME, title_font_size=15, coloraxis_showscale=False)
            st.plotly_chart(fig_dep, use_container_width=True)

        st.plotly_chart(special_requests_chart(fdf), use_container_width=True)

        # Cancellation by hotel type
        hotel_cancel = (
            fdf.groupby("hotel_type")["is_canceled"]
               .agg(["sum","count"])
               .reset_index()
        )
        hotel_cancel["cancel_rate"] = (hotel_cancel["sum"] / hotel_cancel["count"] * 100).round(1)
        c1, c2 = st.columns(2)
        with c1:
            fig_hc = px.bar(
                hotel_cancel, x="hotel_type", y="cancel_rate",
                title="Cancellation Rate by Hotel Type (%)",
                color="hotel_type", color_discrete_sequence=["#3b9eff","#ffb547"],
                labels={"hotel_type":"","cancel_rate":"Rate (%)"},
            )
            fig_hc.update_layout(**PLOTLY_THEME, title_font_size=15, showlegend=False)
            st.plotly_chart(fig_hc, use_container_width=True)
        with c2:
            # Previous cancellations heatmap
            prev = (
                fdf.groupby(["arrival_date_month","is_canceled"], observed=True)
                   .size()
                   .unstack(fill_value=0)
                   .reindex(columns=[0, 1], fill_value=0)
                   .reset_index()
            )
            if 1 in prev.columns:
                prev["cancel_rate"] = prev[1] / (prev[0] + prev[1]).replace(0, 1) * 100
                fig_prev = px.bar(
                    prev, x="arrival_date_month", y="cancel_rate",
                    title="Monthly Cancellation Rate (%)",
                    labels={"arrival_date_month":"Month","cancel_rate":"Rate (%)"},
                    color="cancel_rate",
                    color_continuous_scale=[[0,"#52c87e"],[0.5,"#ffb547"],[1,"#ff6b6b"]],
                )
                fig_prev.update_layout(**PLOTLY_THEME, title_font_size=15, coloraxis_showscale=False)
                st.plotly_chart(fig_prev, use_container_width=True)

    # ── Tab 3: Revenue & ADR ──────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-title">Revenue & ADR Analysis</div>',
                    unsafe_allow_html=True)

        st.plotly_chart(adr_trend_chart(fdf), use_container_width=True)
        st.plotly_chart(revenue_by_hotel_chart(fdf), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            # ADR distribution
            adr_clean = fdf[(fdf["adr"] > 0) & (fdf["adr"] < 1000)]
            fig_adr_dist = px.histogram(
                adr_clean, x="adr", color="hotel_type", nbins=60,
                title="ADR Distribution by Hotel Type",
                labels={"adr": "ADR (₹)", "hotel_type": "Hotel Type"},
                color_discrete_sequence=["#3b9eff", "#ffb547"],
                barmode="overlay", opacity=0.75,
            )
            fig_adr_dist.update_layout(**PLOTLY_THEME, title_font_size=15,
                                       legend=dict(orientation="h", y=1.05))
            st.plotly_chart(fig_adr_dist, use_container_width=True)

        with c2:
            # Top countries by bookings
            top_countries = (
                fdf["country"].value_counts().head(10).reset_index()
            )
            top_countries.columns = ["country", "count"]
            fig_country = px.bar(
                top_countries, x="count", y="country", orientation="h",
                title="Top 10 Source Countries",
                labels={"count": "Bookings", "country": ""},
                color="count",
                color_continuous_scale=[[0,"#1a3a5c"],[1,"#3b9eff"]],
            )
            fig_country.update_layout(**PLOTLY_THEME, title_font_size=15,
                                      coloraxis_showscale=False)
            st.plotly_chart(fig_country, use_container_width=True)

    # ── Tab 4: Predict ────────────────────────────────────────────────────
    with tab4:
        render_prediction_tab(df)  # Always use full df for model training

    # ── Tab 5: Insights ───────────────────────────────────────────────────
    with tab5:
        render_insights()

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="footer">
        🏨 Hotel Demand Intelligence Dashboard &nbsp;|&nbsp;
        Built with Streamlit &amp; Plotly &nbsp;|&nbsp;
        Dataset: Hotel Booking Demand (2024) &nbsp;|&nbsp;
        Model: Gradient Boosting Classifier
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
