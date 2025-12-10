from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

def make_daily():
    import os
    import pandas as pd
    RAW = "/mnt/d/MLOps_Portfolio/data/retail_sales.csv"
    CLEAN = "/mnt/d/MLOps_Portfolio/data/daily_clean.csv"
    
    df = pd.read_csv(RAW, parse_dates=['Date'])
    daily = df.groupby('Date')['TotalPrice'].sum().reset_index()
    daily.columns = ['ds', 'y']
    os.makedirs("/mnt/d/MLOps_Portfolio/data", exist_ok=True)
    daily.to_csv(CLEAN, index=False)
    print("make_daily DONE - daily_clean.csv created")

def forecast_only():
    import os
    import pandas as pd
    from prophet import Prophet
    
    CLEAN = "/mnt/d/MLOps_Portfolio/data/daily_clean.csv"
    OUT = "/mnt/d/MLOps_Portfolio/forecast/output/forecast_latest.csv"
    
    df = pd.read_csv(CLEAN, parse_dates=['ds'])
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    
    os.makedirs("/mnt/d/MLOps_Portfolio/forecast/output", exist_ok=True)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(OUT, index=False)
    
    print("forecast_only DONE - forecast_latest.csv saved")

with DAG(
    dag_id="retail_sales_forecasting_pipeline",
    schedule=None,                     # ← turned off schedule so no confusion
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["retail"],
) as dag:
    t1 = PythonOperator(task_id="make_daily", python_callable=make_daily)
    t2 = PythonOperator(task_id="forecast", python_callable=forecast_only)
    t1 >> t2