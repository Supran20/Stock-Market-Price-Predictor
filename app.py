# ===== IMPORTS =====
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# ===== LOAD MODEL =====
model = load_model('D:/Machine Learning/Stock_Price_Prediction/Stock Prediction Model.keras')

# ===== STREAMLIT UI =====
st.title('📈 Stock Market Price Predictor')

stock_symbol = st.text_input('Enter Stock Symbol (e.g. AAPL, GOOG, TSLA):', 'GOOG')
start_date = '2014-01-01'
end_date = '2024-12-31'

# ===== DOWNLOAD DATA =====
data = yf.download(stock_symbol, start=start_date, end=end_date)
st.subheader('📊 Raw Stock Data')
st.write(data)

# ===== SPLIT DATA =====
train_data = pd.DataFrame(data['Close'][0:int(len(data)*0.80)])
test_data = pd.DataFrame(data['Close'][int(len(data)*0.80):])

# ===== SCALE TEST DATA =====
scaler = MinMaxScaler(feature_range=(0, 1))
last_100_days = train_data.tail(100)
test_full = pd.concat([last_100_days, test_data], ignore_index=True)
test_scaled = scaler.fit_transform(test_full)

# ===== MOVING AVERAGES VISUALIZATION =====
st.subheader('📉 Price vs 50-Day Moving Average')
ma_50 = data['Close'].rolling(window=50).mean()
fig1 = plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label='Close Price', color='green')
plt.plot(ma_50, label='MA 50 Days', color='red')
plt.legend()
st.pyplot(fig1)

st.subheader('📉 Price vs 50-Day & 100-Day Moving Averages')
ma_100 = data['Close'].rolling(window=100).mean()
fig2 = plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label='Close Price', color='green')
plt.plot(ma_50, label='MA 50 Days', color='red')
plt.plot(ma_100, label='MA 100 Days', color='blue')
plt.legend()
st.pyplot(fig2)

st.subheader('📉 Price vs 100-Day & 200-Day Moving Averages')
ma_200 = data['Close'].rolling(window=200).mean()
fig3 = plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label='Close Price', color='green')
plt.plot(ma_100, label='MA 100 Days', color='red')
plt.plot(ma_200, label='MA 200 Days', color='blue')
plt.legend()
st.pyplot(fig3)

# ===== SEQUENCE GENERATION FOR TEST DATA =====
x_test = []
y_test = []

for i in range(100, test_scaled.shape[0]):
    x_test.append(test_scaled[i-100:i])
    y_test.append(test_scaled[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)

# ===== PREDICTION =====
predictions = model.predict(x_test)

# ===== INVERSE SCALING =====
scale_factor = 1 / scaler.scale_[0]
predictions = predictions * scale_factor
y_test = y_test * scale_factor

# ===== FINAL COMPARISON PLOT =====
st.subheader('📈 Predicted vs Actual Stock Prices')
fig4 = plt.figure(figsize=(10, 6))
plt.plot(predictions, label='Predicted Price', color='red')
plt.plot(y_test, label='Actual Price', color='green')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title(f'{stock_symbol.upper()} Price Prediction')
plt.legend()
st.pyplot(fig4)
