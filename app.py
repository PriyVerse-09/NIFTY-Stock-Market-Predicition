import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from pathlib import Path
from datetime import datetime, timedelta
from nsepython import nse_holidays
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Hybrid ARIMA–LSTM NIFTY Predictor", layout="wide", page_icon="📈")

BASE_DIR = Path(__file__).resolve().parent
ARIMA_PATH = BASE_DIR / "arima_model.pkl"
LSTM_KERAS_PATH = BASE_DIR / "lstm_model.keras"
LSTM_H5_PATH = BASE_DIR / "lstm_model.h5"
SCALER_PATH = BASE_DIR / "residual_scaler.pkl"

@st.cache_resource
def load_models():
    missing = [p.name for p in (ARIMA_PATH, SCALER_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing model artifact(s): " + ", ".join(missing) + ". Run training.py and place the generated artifacts in the repository root.")

    arima = joblib.load(ARIMA_PATH)
    scaler = joblib.load(SCALER_PATH)

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
        raise FileNotFoundError("Missing LSTM model artifact: lstm_model.keras or lstm_model.h5. Run training.py and place the generated artifact in the repository root.")

    return arima, lstm, scaler

arima_model, lstm_model, scaler = load_models()

@st.cache_data
def fetch_data():
    end = datetime.today()
    start = end - timedelta(days=365 * 4)
    df = yf.download("^NSEI", start=start, end=end, interval="1d")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].dropna()
    if df.empty:
        raise ValueError("No NIFTY data was downloaded from Yahoo Finance.")
    return df

df = fetch_data()

st.title("Hybrid ARIMA–LSTM NIFTY Prediction Dashboard")
st.caption("Short-term forecasting using ARIMA + LSTM residual correction")

holidays = nse_holidays()

def next_trading_day(d):
    d += timedelta(days=1)
    while d.weekday() >= 5 or d.strftime("%Y-%m-%d") in holidays:
        d += timedelta(days=1)
    return d

# -----------------------------------------------------------------------------
# ARIMA FORECAST
# -----------------------------------------------------------------------------
# The serialized Statsmodels result can contain an index that is not usable by
# the version of Statsmodels running on Streamlit Cloud. Calling forecast() on
# that object therefore raises: "ValueError: No supported index is available."
#
# Fix: reuse the saved ARIMA ORDER, but fit a fresh model on the current NIFTY
# values using an explicit RangeIndex. A RangeIndex is always supported for a
# one-step forecast, so this avoids the deployment-specific index failure.
current_series = df["Close"].astype(float)

arima_order = getattr(getattr(arima_model, "model", None), "order", None)
if arima_order is None:
    arima_order = getattr(arima_model, "model_orders", {}).get("arima", None)

if arima_order is None:
    raise RuntimeError("Could not read the ARIMA order from arima_model.pkl. Please regenerate the model with training.py.")

# Explicit integer index: forecast() will never need a missing DatetimeIndex.
arima_input = pd.Series(
    current_series.to_numpy(dtype=float),
    index=pd.RangeIndex(start=0, stop=len(current_series), step=1),
    name="Close"
)

try:
    arima_current = ARIMA(arima_input, order=arima_order).fit()
    arima_next = float(np.asarray(arima_current.forecast(steps=1)).reshape(-1)[0])
except Exception as exc:
    raise RuntimeError(
        f"ARIMA forecasting failed for order {arima_order}: {exc}"
    ) from exc

# -----------------------------------------------------------------------------
# LSTM RESIDUAL CORRECTION
# -----------------------------------------------------------------------------
# Match the residual definition used by the current ARIMA fit.
arima_fitted_values = np.asarray(arima_current.fittedvalues).reshape(-1)
min_len = min(len(current_series), len(arima_fitted_values))

actual_values = current_series.to_numpy(dtype=float)[-min_len:]
fitted_values = arima_fitted_values[-min_len:]
residuals = actual_values - fitted_values
residuals = residuals.reshape(-1, 1)

# The LSTM was trained with a 5-day residual window.
time_steps = 5
if len(residuals) < time_steps:
    raise ValueError(
        f"Not enough residual observations for the LSTM window. "
        f"Required {time_steps}, found {len(residuals)}."
    )

scaled_resid = scaler.transform(residuals)
X_last = scaled_resid[-time_steps:].reshape(1, time_steps, 1)

lstm_next_scaled = float(lstm_model.predict(X_last, verbose=0)[0][0])
lstm_next_resid = float(
    scaler.inverse_transform(np.array([[lstm_next_scaled]]))[0][0]
)

next_pred = float(arima_next + lstm_next_resid)

last_date = df.index[-1].date()
next_date = next_trading_day(last_date)
prev_close = float(current_series.iloc[-1])

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

past_dates = list(df.index[-past_days:].date)
past_values = [float(x) for x in current_series.iloc[-past_days:]]
plot_dates = past_dates + [next_date]
plot_values = past_values + [next_pred]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(plot_dates, plot_values, marker="o", linewidth=2)
ax.scatter(
    plot_dates[-1],
    plot_values[-1],
    color="red",
    s=80,
    label="Prediction"
)

import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.grid(True)
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)
st.markdown("---")
