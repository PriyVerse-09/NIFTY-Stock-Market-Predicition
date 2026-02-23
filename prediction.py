#  NEXT DAY PREDICTION
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from datetime import datetime, timedelta
from nsepython import nse_holidays
from tensorflow.keras.models import load_model
from datetime import timedelta
from nsepython import nse_holidays
import numpy as np
from dateutil.relativedelta import relativedelta

nifty = yf.Ticker("^NSEI")

# Define rolling window
END_DATE = datetime.today()
START_DATE = END_DATE - relativedelta(years=4)

# Fetch data
df = nifty.history(
    start=START_DATE.strftime("%Y-%m-%d"),
    end=END_DATE.strftime("%Y-%m-%d"),
    interval="1d"
)

arima_model = joblib.load("arima_model.pkl")
lstm_model = load_model("lstm_model.h5", compile=False)
scaler = joblib.load("residual_scaler.pkl")

full_close = df['Close'].values

arima_fitted = np.array(arima_model.fittedvalues).flatten()

# align lengths
min_len_full = min(len(full_close), len(arima_fitted))
close_aligned = full_close[-min_len_full:]
fitted_aligned = arima_fitted[-min_len_full:]

# compute full residuals
residuals_full = close_aligned - fitted_aligned
residuals_full = residuals_full.reshape(-1, 1)

# scale using saved scaler
scaled_full = scaler.transform(residuals_full)

# prepare last window for LSTM
time_steps = 5
X_last = scaled_full[-time_steps:].reshape(1, time_steps, 1)

# LSTM predicts next residual
lstm_next_scaled = lstm_model.predict(X_last, verbose=0)[0][0]
lstm_next_resid  = scaler.inverse_transform([[lstm_next_scaled]])[0][0]

# ARIMA one-step forecast
arima_next = float(arima_model.forecast(steps=1).iloc[0])

# hybrid forecast
next_pred = arima_next + lstm_next_resid

# NEXT TRADING DAY CALCULATION

holidays = nse_holidays()

def next_trading_day(last_date):
    nxt = last_date + timedelta(days=1)
    while True:
        s = nxt.strftime("%Y-%m-%d")
        if nxt.weekday() >= 5:  # weekend
            nxt += timedelta(days=1)
            continue
        if s in holidays:       # NSE holiday
            nxt += timedelta(days=1)
            continue
        return nxt

last_day = df.index[-1].date()
next_day = next_trading_day(last_day)

prev_close = full_close[-1]

print("\n**************** NEXT DAY PREDICTION ****************")
print("Last Trading Day:", last_day)
print("Next Trading Day:", next_day)
print(f"Previous Close: {prev_close:.2f}")
print(f"Predicted Close ({next_day}): {next_pred:.2f}")
print(f"Expected Change: {next_pred - prev_close:.2f} points")