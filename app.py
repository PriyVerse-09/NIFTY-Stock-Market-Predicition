import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import load_model
from pathlib import Path
from datetime import datetime, timedelta
from nsepython import nse_holidays
import matplotlib.pyplot as plt

# PAGE CONFIG
st.set_page_config(
    page_title="Hybrid ARIMA–LSTM NIFTY Predictor",
    layout="wide",
    page_icon="📈"
)

# MODEL PATHS
BASE_DIR = Path(__file__).resolve().parent
ARIMA_PATH = BASE_DIR / "arima_model.pkl"
LSTM_KERAS_PATH = BASE_DIR / "lstm_model.keras"
LSTM_H5_PATH = BASE_DIR / "lstm_model.h5"
SCALER_PATH = BASE_DIR / "residual_scaler.pkl"

# LOAD MODELS
@st.cache_resource
def load_models():
    missing = [
        str(path.name)
        for path in (ARIMA_PATH, SCALER_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing model artifact(s): " + ", ".join(missing) +
            ". Run training.py and place the generated artifacts in the repository root."
        )

    arima = joblib.load(ARIMA_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Prefer the modern Keras format when available. For the existing legacy
    # H5 artifact, reconstruct the known architecture and load only its weights
    # to avoid Keras H5 config deserialization errors.
    if LSTM_KERAS_PATH.exists():
        lstm = load_model(LSTM_KERAS_PATH, compile=False)
    elif LSTM_H5_PATH.exists():
        lstm = Sequential([
            LSTM(50, input_shape=(5, 1)),
            Dropout(0.2),
            Dense(1)
        ])
        lstm.build((None, 5, 1))
        lstm.load_weights(LSTM_H5_PATH)
    else:
        raise FileNotFoundError(
            "Missing LSTM model artifact: lstm_model.keras or lstm_model.h5. "
            "Run training.py and place the generated artifact in the repository root."
        )

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

# NEXT TRADING DAY
holidays = nse_holidays()

def next_trading_day(d):
    d += timedelta(days=1)
    while d.weekday() >= 5 or d.strftime("%Y-%m-%d") in holidays:
        d += timedelta(days=1)
    return d

# ONE DAY PREDICTION LOGIC
full_close = df['Close'].values
arima_fitted = np.array(arima_model.fittedvalues).flatten()

min_len = min(len(full_close), len(arima_fitted))
residuals = full_close[-min_len:] - arima_fitted[-min_len:]
residuals = residuals.reshape(-1, 1)

scaled_resid = scaler.transform(residuals)

time_steps = 5
if len(scaled_resid) < time_steps:
    raise ValueError(
        f"Not enough residual observations for the LSTM window. "
        f"Required {time_steps}, found {len(scaled_resid)}."
    )

X_last = scaled_resid[-time_steps:].reshape(1, time_steps, 1)

lstm_next_scaled = lstm_model.predict(X_last, verbose=0)[0][0]
lstm_next_resid = float(
    scaler.inverse_transform([[lstm_next_scaled]])[0][0]
)

arima_next = float(arima_model.forecast(steps=1).iloc[0])
next_pred = float(arima_next + lstm_next_resid)

last_date = df.index[-1].date()
next_date = next_trading_day(last_date)
prev_close = float(full_close[-1])

# DISPLAY RESULTS
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
next_pred = float(next_pred)

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

# FOOTER
st.markdown("---")
