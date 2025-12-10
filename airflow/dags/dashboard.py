
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

st.set_page_config(page_title="Rameen Fatima - Retail Forecaster", layout="wide")
st.title("Retail Sales Forecasting System")
st.markdown("### Upload your sales CSV → Get 6 predictions + beautiful plots instantly")

uploaded_file = st.file_uploader("Choose your retail_sales.csv", type=["csv"])

if uploaded_file is not None:

    # ---------------------------
    # FIX 1: Clean column names
    # ---------------------------
    df = pd.read_csv(uploaded_file)
    df.columns = (
        df.columns
        .str.strip()          # remove leading/trailing spaces
        .str.lower()          # lowercase
        .str.replace(" ", "_") # replace spaces with _
    )

    # Debug print (optional)
    # st.write("Cleaned Columns:", df.columns.tolist())

    # ---------------------------
    # FIX 2: Ensure required columns exist
    # ---------------------------
    required = ["unitprice", "quantity", "date"]
    for col in required:
        if col not in df.columns:
            st.error(f"Missing required column: **{col}** \nFound columns: {df.columns.tolist()}")
            st.stop()

    # Convert Date column to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # ---------------------------
    # FIX 3: Create TotalPrice safely
    # ---------------------------
    if 'totalprice' not in df.columns:
        df['totalprice'] = df['quantity'] * df['unitprice']

    # Default optional fields
    df['channel'] = df.get('channel', 'Online')
    df['region'] = df.get('region', 'North')

    # ---------------------------
    # GROUP BY DAILY SALES
    # ---------------------------
    daily = df.groupby('date')['totalprice'].sum().reset_index()
    daily.columns = ['ds', 'y']

    # ---------------------------
    # MODEL TRAINING
    # ---------------------------
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(daily)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)

    # ---------------------------
    # METRICS CARDS
    # ---------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Next Day Sales", f"₹{int(forecast.tail(1)['yhat'].iloc[0]):,}")
    with col2:
        st.metric("Next 7 Days Total", f"₹{int(forecast.tail(7)['yhat'].sum()):,}")
    with col3:
        st.metric("Next 30 Days Total", f"₹{int(forecast.tail(30)['yhat'].sum()):,}")

    # ---------------------------
    # PLOTLY FORECAST FIGURE
    # ---------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily['ds'], y=daily['y'], name="Historical", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Forecast", line=dict(color="#ff7f0e")))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, line=dict(color="lightgray"), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', fillcolor="rgba(255,127,14,0.2)", name="80% Confidence"))
    fig.update_layout(title="30-Day Sales Forecast", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # BAR CHARTS
    # ---------------------------
    col4, col5 = st.columns(2)
    with col4:
        st.bar_chart(df.groupby('channel')['totalprice'].sum())
        st.write("Sales by Channel")
    with col5:
        st.bar_chart(df.groupby('region')['totalprice'].sum())
        st.write("Sales by Region")

    # ---------------------------
    # SUCCESS MESSAGE
    # ---------------------------
    st.success("All 6 predictions ready! ")
