stocks prediction                          
code:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Generate synthetic Netflix stock price data
np.random.seed(0)
dates = pd.date_range(start="2022-01-01", periods=100, freq="D")
prices = np.cumsum(np.random.randn(100) + 5)

data = pd.DataFrame({'Date': dates, 'Close': prices})

# Extract the 'Close' prices and convert them to numpy array
prices = data['Close'].values.reshape(-1, 1)

# Normalize the data to the range [0, 1]
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

# Split the data into training and testing sets
train_size = int(len(prices_scaled) * 0.80)
test_size = len(prices_scaled) - train_size
train_data, test_data = prices_scaled[0:train_size], prices_scaled[train_size:]

# Create sequences for training and testing
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10  # Adjust this for your preference
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Build an LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform the predictions to get real stock prices
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform(y_train)
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform(y_test)

# Calculate RMSE (Root Mean Squared Error) on the test set
rmse = np.sqrt(mean_squared_error(y_test, test_predict))
print("Root Mean Squared Error (RMSE):", rmse)

# Plot the actual vs. predicted stock prices
plt.figure(figsize=(14, 6))
plt.title('Netflix Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.plot(data['Date'][-len(y_test):], y_test, label='Actual Prices', color='blue')
plt.plot(data['Date'][-len(y_test):], test_predict, label='Predicted Prices', color='red')
plt.legend()
plt.show()