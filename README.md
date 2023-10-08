import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load the stock price data (you'll need to replace 'AAPL' with the stock symbol you want)
# Make sure you have a CSV file with columns 'Date' and 'Close' containing historical stock prices.
data = pd.read_csv('stock_data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Plot the historical stock price data
plt.figure(figsize=(12, 6))
plt.title('Stock Price History')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.plot(data['Close'])
plt.show()

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create sequences of data for the LSTM model
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequence = data[i:i + seq_length]
        target = data[i + seq_length]
        sequences.append((sequence, target))
    return np.array(sequences)

seq_length = 10
train_sequences = create_sequences(train_data, seq_length)
test_sequences = create_sequences(test_data, seq_length)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(train_sequences[:, 0], train_sequences[:, 1], epochs=10, batch_size=32)

# Evaluate the model on the test data
test_loss = model.evaluate(test_sequences[:, 0], test_sequences[:, 1])
print(f'Test Loss: {test_loss}')

# Make predictions
predictions = model.predict(test_sequences[:, 0])
predictions = scaler.inverse_transform(predictions)

# Plot the actual vs. predicted stock prices
plt.figure(figsize=(12, 6))
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.plot(data.index[train_size+seq_length:], data['Close'][train_size+seq_length:], label='Actual Price')
plt.plot(data.index[train_size+seq_length:], predictions, label='Predicted Price')
plt.legend()
plt.show()
