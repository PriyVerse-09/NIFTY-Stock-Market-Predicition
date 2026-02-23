# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import itertools

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import yfinance as yf
import joblib
import warnings
warnings.filterwarnings('ignore')

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
print(df.head(), df.tail())

result = adfuller(df['Close'])
print("ADF:", result[0], "| p-value:", result[1])

df_diff = df['Close'].diff().dropna()

result_diff = adfuller(df_diff)
print("After differencing → p-value:", result_diff[1])

split_idx = int(len(df) * 0.7)
train = df['Close'][:split_idx]
test = df['Close'][split_idx:]

print("Train:", len(train), "Test:", len(test))

p = q = range(0, 3)
d = 1
pdq = list(itertools.product(p, [d], q))

best_aic = np.inf
best_order, best_model = None, None

for order in pdq:
    try:
        model = ARIMA(train, order=order)
        results = model.fit()
        if results.aic < best_aic:
            best_aic = results.aic
            best_order = order
            best_model = results
    except:
        continue

print("Best ARIMA order:", best_order, "AIC:", best_aic)
arima_model = best_model

start_idx = len(train)
end_idx   = len(train) + len(test) - 1

arima_pred = arima_model.get_prediction(start=start_idx, end=end_idx)
arima_pred_test = arima_pred.predicted_mean

# align index with actual test dates
arima_pred_test.index = test.index

residuals = test.values - arima_pred_test.values
residuals = residuals.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_residuals = scaler.fit_transform(residuals)

def create_sequences(data, time_steps=5):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

time_steps = 5
X_lstm, y_lstm = create_sequences(scaled_residuals, time_steps)
X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))

model = Sequential([
    LSTM(50, input_shape=(time_steps,1)),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

history = model.fit(
    X_lstm, y_lstm,
    epochs=50,
    batch_size=16,
    verbose=1
)

# LSTM prediction (scaled)
lstm_pred_scaled = model.predict(X_lstm)

# Inverse transform to get real residuals
lstm_pred_residuals = scaler.inverse_transform(lstm_pred_scaled)
lstm_pred_residuals = lstm_pred_residuals.flatten()

# Compute min length
min_len = min(len(arima_pred_test.values), len(lstm_pred_residuals))

# Align both arrays to SAME length
arima_pred_aligned = arima_pred_test.values[-min_len:]
lstm_resid_aligned = lstm_pred_residuals[-min_len:]

# # Final prediction
# final_pred = arima_pred_aligned + lstm_resid_aligned

# actual = test.iloc[-min_len:].values

# rmse = np.sqrt(mean_squared_error(actual, final_pred))
# mae  = mean_absolute_error(actual, final_pred)
# r2   = r2_score(actual, final_pred)

# print(f"Hybrid ARIMA-LSTM RMSE: {rmse:.2f}")
# print(f"Hybrid ARIMA-LSTM MAE: {mae:.2f}")
# print(f"Hybrid ARIMA-LSTM R2 Score: {r2:.2f}")

# plt.figure(figsize=(12,5))
# plt.plot(test.index[-min_len:], actual, label='Actual')
# plt.plot(test.index[-min_len:], final_pred, label='Hybrid Prediction')
# plt.legend()
# plt.grid(True)
# plt.show()

joblib.dump(arima_model, "arima_model.pkl")
model.save("lstm_model.h5")
joblib.dump(scaler, "residual_scaler.pkl")

print("Models saved successfully.")
