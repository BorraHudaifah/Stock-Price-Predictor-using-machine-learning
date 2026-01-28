import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Step 1: Data Collection
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data[['Close']]  # Focus on closing prices

ticker = 'AAPL'
start_date = '2015-01-01'
end_date = '2023-01-01'
data = fetch_data(ticker, start_date, end_date)
print(f"Data shape: {data.shape}")
print(data.head())

# Step 2: Preprocessing
# Add features for non-LSTM models
data['Lag1'] = data['Close'].shift(1)
data['Lag2'] = data['Close'].shift(2)
data['Lag3'] = data['Close'].shift(3)
data['MA7'] = data['Close'].rolling(window=7).mean()  # 7-day moving average
data.dropna(inplace=True)  # Remove NaNs

# Features and target
features = ['Lag1', 'Lag2', 'Lag3', 'MA7']
X = data[features]
y = data['Close']

# Train-test split (chronological)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 3: Model Training and Evaluation - Basic Models
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    print(f"{model_name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    return predictions

# Linear Regression
lr_model = LinearRegression()
lr_preds = evaluate_model(lr_model, X_train, y_train, X_test, y_test, "Linear Regression")

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_preds = evaluate_model(rf_model, X_train, y_train, X_test, y_test, "Random Forest")

# XGBoost
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_preds = evaluate_model(xgb_model, X_train, y_train, X_test, y_test, "XGBoost")

# Step 4: LSTM Model
# Prepare data for LSTM (sequences)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close']].values)

seq_length = 60  # Use last 60 days to predict next
X_lstm, y_lstm = [], []
for i in range(seq_length, len(scaled_data)):
    X_lstm.append(scaled_data[i-seq_length:i, 0])
    y_lstm.append(scaled_data[i, 0])
X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))

# Split LSTM data
train_size_lstm = int(0.8 * len(X_lstm))
X_train_lstm, X_test_lstm = X_lstm[:train_size_lstm], X_lstm[train_size_lstm:]
y_train_lstm, y_test_lstm = y_lstm[:train_size_lstm], y_lstm[train_size_lstm:]

# Build LSTM model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, verbose=1)

# Predict and evaluate
lstm_preds_scaled = lstm_model.predict(X_test_lstm)
lstm_preds = scaler.inverse_transform(lstm_preds_scaled)
y_test_actual = scaler.inverse_transform(y_test_lstm.reshape(-1, 1))
rmse_lstm = np.sqrt(mean_squared_error(y_test_actual, lstm_preds))
mae_lstm = mean_absolute_error(y_test_actual, lstm_preds)
print(f"LSTM - RMSE: {rmse_lstm:.2f}, MAE: {mae_lstm:.2f}")

# Step 5: Visualization
plt.figure(figsize=(14, 7))
plt.plot(y_test.values, label='Actual Prices', color='blue')
plt.plot(lr_preds, label='Linear Regression Predictions', color='red', linestyle='--')
plt.plot(rf_preds, label='Random Forest Predictions', color='green', linestyle='--')
plt.plot(xgb_preds, label='XGBoost Predictions', color='orange', linestyle='--')
plt.plot(lstm_preds, label='LSTM Predictions', color='purple', linestyle='--')
plt.title(f'Stock Price Predictions for {ticker}')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()

# Step 6: Prediction for Next Day (Example)
# For basic models, use last available features
last_features = X.iloc[-1].values.reshape(1, -1)
lr_next = lr_model.predict(last_features)[0]
rf_next = rf_model.predict(last_features)[0]
xgb_next = xgb_model.predict(last_features)[0]

# For LSTM, use last 60 days
last_60 = scaled_data[-60:].reshape(1, 60, 1)
lstm_next_scaled = lstm_model.predict(last_60)
lstm_next = scaler.inverse_transform(lstm_next_scaled)[0][0]

print(f"\nNext Day Predictions for {ticker}:")
print(f"Linear Regression: ${lr_next:.2f}")
print(f"Random Forest: ${rf_next:.2f}")
print(f"XGBoost: ${xgb_next:.2f}")
print(f"LSTM: ${lstm_next:.2f}")