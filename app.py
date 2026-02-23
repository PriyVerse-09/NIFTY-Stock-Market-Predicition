import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
from nsepython import nse_holidays
import matplotlib.pyplot as plt

# PAGE CONFIG 
st.set_page_config(
    page_title="Hybrid ARIMA–LSTM NIFTY Predictor",
    layout="wide",
    page_icon="📈"
)

#  LOAD MODELS 
@st.cache_resource
def load_models():
    arima = joblib.load("arima_model.pkl")
    lstm = load_model("lstm_model.h5", compile=False)
    scaler = joblib.load("residual_scaler.pkl")
    return arima, lstm, scaler

arima_model, lstm_model, scaler = load_models()

# DATA FETCH
@st.cache_data
def fetch_data():
    end = datetime.today()
    start = end - timedelta(days=365 * 4)

    df = yf.download("^NSEI", start=start, end=end, interval="1d")


    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Close']].dropna()
    return df

df = fetch_data()

# TITLE 
st.title("Hybrid ARIMA–LSTM NIFTY Prediction Dashboard")
st.caption("Short-term forecasting using ARIMA + LSTM residual correction")

#  NEXT TRADING DAY 
holidays = nse_holidays()

def next_trading_day(d):
    d += timedelta(days=1)
    while d.weekday() >= 5 or d.strftime("%Y-%m-%d") in holidays:
        d += timedelta(days=1)
    return d


#  ONE DAY PREDICTION LOGIC 

# Full close series
full_close = df['Close'].values

# ARIMA fitted values
arima_fitted = np.array(arima_model.fittedvalues).flatten()

# Align lengths
min_len = min(len(full_close), len(arima_fitted))
residuals = full_close[-min_len:] - arima_fitted[-min_len:]
residuals = residuals.reshape(-1, 1)

# Scale residuals
scaled_resid = scaler.transform(residuals)

# Prepare last LSTM window
time_steps = 5
X_last = scaled_resid[-time_steps:].reshape(1, time_steps, 1)

# Predict next residual (FORCE FLOAT)
lstm_next_scaled = lstm_model.predict(X_last, verbose=0)[0][0]
lstm_next_resid = float(
    scaler.inverse_transform([[lstm_next_scaled]])[0][0]
)

# ARIMA next forecast (FORCE FLOAT)
arima_next = float(arima_model.forecast(steps=1).iloc[0])

# Final hybrid prediction
next_pred = float(arima_next + lstm_next_resid)

# Dates
last_date = df.index[-1].date()
next_date = next_trading_day(last_date)

# Previous close
prev_close = float(full_close[-1])


#  DISPLAY RESULTS 
summary_df = pd.DataFrame({
    "Date": [last_date, next_date],
    "Close Value": [round(prev_close, 2), round(next_pred, 2)]
})

st.subheader("📅 Prediction Summary")
st.dataframe(summary_df, use_container_width=True)


# FORECAST CURVE 
st.subheader("📈 Forecast Curve")

past_days = st.selectbox(
    "Show past days",
    options=[1, 2, 3, 4],
    index=3
)
prev_close = float(df['Close'].iloc[-1])
next_pred = float(next_pred)   # VERY IMPORTANT

past_dates = list(df.index[-past_days:].date)
past_values = [float(x) for x in df['Close'].iloc[-past_days:]]

plot_dates = past_dates + [next_date]
plot_values = past_values + [next_pred]


fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(
    plot_dates,
    plot_values,
    marker='o',
    linewidth=2
)

# Highlight prediction
ax.scatter(
    plot_dates[-1],
    plot_values[-1],
    color='red',
    s=80,
    label="Prediction"
)

import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.grid(True)
ax.legend()

plt.xticks(rotation=45)
st.pyplot(fig)

# ------------------ FOOTER ------------------
st.markdown("---")



