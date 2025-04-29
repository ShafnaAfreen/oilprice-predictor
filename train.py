import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# 1️⃣ Download Crude Oil Price Data
print("Downloading oil price data...")
df = yf.download('CL=F', start='2015-01-01', end='2025-01-01')
df = df[['Close']].dropna()

# 2️⃣ Normalize Data (Scaling between 0 and 1)
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)

# 3️⃣ Create Time-Series Sequences for Training
def create_sequences(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

time_steps = 10  # Use last 10 days to predict the next day
X, y = create_sequences(df_scaled, time_steps)

# 4️⃣ Split Data (80% Train, 20% Test)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5️⃣ Build LSTM Model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(time_steps, 1)),
    LSTM(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# 6️⃣ Train Model
print("Training LSTM model...")
model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)

# 7️⃣ Predict Future Prices
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)  # Convert back to original scale
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# 8️⃣ Plot Actual vs Predicted Prices
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(y_pred, label='Predicted Prices', linestyle='dashed', color='red')
plt.legend()
plt.xlabel("Days")
plt.ylabel("Oil Price (USD)")
plt.title("LSTM Oil Price Prediction")
plt.show()

# 9️⃣ Predict Next Day's Price
latest_data = df_scaled[-time_steps:].reshape(1, time_steps, 1)  # Reshape for LSTM input
predicted_price = model.predict(latest_data)
predicted_price = scaler.inverse_transform(predicted_price)[0][0]
print(f"Predicted Oil Price for Tomorrow: ${predicted_price:.2f}")

# Assuming 'y_test' are actual prices and 'y_pred' are model predictions
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")  # Closer to 1 means better prediction
