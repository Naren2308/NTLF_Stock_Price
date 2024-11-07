import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from flask import Flask, render_template, jsonify, request
import os
import json

app = Flask(__name__)

# Route for serving the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for processing and serving predictions
@app.route('/get_predictions', methods=['GET'])
def get_predictions():
    # Load the data (this would be replaced with your actual data processing)
    data = pd.read_csv('NFLX.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data[['Close']]

    # Feature Engineering (you might have more feature engineering here)
    data['5_MA'] = data['Close'].rolling(window=5).mean()
    data['30_MA'] = data['Close'].rolling(window=30).mean()
    data.dropna(inplace=True)

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # Prepare data for LSTM
    lookback = 90
    X = []
    y = []
    for i in range(lookback, len(data_scaled)):
        X.append(data_scaled[i-lookback:i])
        y.append(data_scaled[i, 0])
    X = np.array(X)
    y = np.array(y)

    # LSTM Model
    model = Sequential()
    model.add(LSTM(units=60, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=60, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=50, batch_size=32)

    # Make Predictions
    predictions = model.predict(X)

    # Inverse transform predictions
    predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], data_scaled.shape[1]-1))), axis=1))[:, 0]

    # Prepare prediction dates
    prediction_dates = data.index[lookback:].tolist()

    # Create a dictionary with the prediction dates and values
    prediction_data = {
        'dates': [date.strftime('%Y-%m-%d') for date in prediction_dates],
        'predictions': predictions.tolist()
    }

    # Return the data as a JSON response
    return jsonify(prediction_data)

if __name__ == '__main__':
    app.run(debug=True)
