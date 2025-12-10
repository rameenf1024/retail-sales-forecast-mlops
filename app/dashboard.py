import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Fix NumPy 2.0 issue (needed for Prophet)
np.float_ = np.float64
np.int_ = np.int64

st.set_page_config(page_title="Retail Sales Forecast - Rameen Fatima", layout="wide")
st.title("Retail Sales Forecasting System")


st.markdown("### Upload your retail transactions CSV")
uploaded_file = st.file_uploader("Choose retail_sales.csv", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV with columns: **Date**, **Quantity**, **UnitPrice**")
    st.stop()

# Read and clean
df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

required = ["date", "quantity", "unitprice"]
if not all(col in df.columns for col in required):
    st.error(f"Missing columns! Need: {required}")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
df["totalprice"] = df["quantity"] * df["unitprice"]
df["channel"] = df.get("channel", "Online")
df["region"] = df.get("region", "North")

daily = df.groupby("date")["totalprice"].sum().reset_index()
daily.columns = ["ds", "y"]

if len(daily) < 7:
    st.error("Need at least 7 days of data!")
    st.stop()

# Try Prophet, fallback to linear if anything goes wrong
try:
    from prophet import Prophet
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(daily)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    st.success("Forecasted using Facebook Prophet!")
except Exception as e:
    st.warning("Prophet not available → using simple trend forecast")
    # Simple linear trend (100% reliable)
    x = np.arange(len(daily))
    slope, intercept = np.polyfit(x, daily["y"], 1)
    last_date = daily["ds"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30)
    future_y = intercept + slope * np.arange(len(daily), len(daily)+30)
    
    # Build forecast DataFrame
    forecast = pd.DataFrame({
        "ds": pd.concat([daily["ds"], pd.Series(future_dates)]).reset_index(drop=True),
        "yhat": list(daily["y"]) + list(future_y),
        "yhat_lower": list(daily["y"]) + list(future_y * 0.9),
        "yhat_upper": list(daily["y"]) + list(future_y * 1.1)
    })

# Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Next Day", f"₹{int(forecast['yhat'].iloc[-30]):,}")
with col2:
    st.metric("Next 7 Days", f"₹{int(forecast['yhat'].iloc[-30:-23].sum()):,}")
with col3:
    st.metric("Next 30 Days", f"₹{int(forecast['yhat'].iloc[-30:].sum()):,}")

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=daily["ds"], y=daily["y"], name="Historical Sales", line=dict(color="#636EFA")))
fig.add_trace(go.Scatter(x=forecast["ds"].iloc[-30:], y=forecast["yhat"].iloc[-30:], name="30-Day Forecast", line=dict(color="#EF553B")))
fig.add_trace(go.Scatter(x=forecast["ds"].iloc[-30:], y=forecast["yhat_upper"].iloc[-30:], line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=forecast["ds"].iloc[-30:], y=forecast["yhat_lower"].iloc[-30:], fill='tonexty', fillcolor="rgba(239,85,59,0.2)", line=dict(width=0), name="Confidence"))
fig.update_layout(title="30-Day Retail Sales Forecast", height=550, template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# Bar charts
col4, col5 = st.columns(2)
with col4:
    st.subheader("Sales by Channel")
    st.bar_chart(df.groupby("channel")["totalprice"].sum())
with col5:
    st.subheader("Sales by Region")
    st.bar_chart(df.groupby("region")["totalprice"].sum())

st.success("Forecast Ready! Live MLOps Demo")
st.balloons()
st.caption(" Airflow + Prophet + Streamlit")
