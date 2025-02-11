import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Step 1: Fetch Stock Data
def get_stock_data(ticker, start='2015-01-01', end='2024-01-01'):
    data = yf.download(ticker, start=start, end=end)
    return data['Close']

# Step 2: Prepare Data for LSTM Model
def prepare_data(data, time_steps=50):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM
    
    return X, y, scaler

# Step 3: Build LSTM Model
def build_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == "__main__":
    ticker = 'AAPL'  # Example: Apple stock
    stock_data = get_stock_data(ticker)
    X, y, scaler = prepare_data(stock_data)
    
    model = build_model((X.shape[1], 1))
    model.fit(X, y, epochs=20, batch_size=32)
    
    # Predict future stock price
    predicted_price = model.predict(X[-1].reshape(1, X.shape[1], 1))
    predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))
    
    print(f"Predicted Next Day Price for {ticker}: ${predicted_price[0][0]:.2f}")
    
    # Plot stock prices
    plt.figure(figsize=(10,5))
    plt.plot(stock_data, label='Actual Price')
    plt.title(f'{ticker} Stock Price History')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
