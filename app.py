from flask import Flask, render_template, request, send_from_directory
import os
import datetime
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xgboost as xgb

app = Flask(__name__)
app.config['PREDICTIONS_FOLDER'] = 'static/predictions'

# Load models and scaler
lstm_model = load_model('netflix_lstm_model.h5')
xgb_model = xgb.XGBRegressor()
xgb_model.load_model('netflix_xgb_model.json')
scaler = joblib.load('scaler.pkl')

# Load and preprocess dataset
data = pd.read_csv('NFLX_dataset.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data['5_MA'] = data['Close'].rolling(window=5).mean()
data['30_MA'] = data['Close'].rolling(window=30).mean()
data['60_MA'] = data['Close'].rolling(window=60).mean()
data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
data['Signal_Line'] = data['MACD'].ewm(span=9).mean()
data['RSI'] = 100 - (100 / (1 + data['Close'].diff().apply(lambda x: max(x, 0)).rolling(window=14).mean() / data['Close'].diff().apply(lambda x: abs(x)).rolling(window=14).mean()))
data['Volatility'] = data['Close'].rolling(window=5).std()
data['Returns'] = data['Close'].pct_change()
data.dropna(inplace=True)

required_features = ['Close', '5_MA', '30_MA', '60_MA', 'MACD', 'Signal_Line', 'RSI', 'Volatility', 'Returns']
data_scaled = scaler.transform(data[required_features])

# Prediction function
def predict_custom_dates(custom_dates, required_features):
    custom_dates = [datetime.datetime.strptime(date, '%d-%m-%Y') for date in custom_dates]
    last_lookback_data = data_scaled[-120:].reshape(1, 120, len(required_features))

    lstm_predictions = []
    xgb_predictions = []
    for _ in custom_dates:
        lstm_pred = lstm_model.predict(last_lookback_data)
        lstm_predictions.append(lstm_pred[0, 0])

        xgb_input = last_lookback_data.reshape(1, -1)
        xgb_pred = xgb_model.predict(xgb_input)
        xgb_predictions.append(xgb_pred[0])

        combined_pred = (lstm_pred[0, 0] + xgb_pred[0]) / 2
        last_lookback_data = np.append(last_lookback_data[:, 1:, :], [[[combined_pred]]], axis=1)

    hybrid_predictions = [(lstm + xgb) / 2 for lstm, xgb in zip(lstm_predictions, xgb_predictions)]
    hybrid_predictions = scaler.inverse_transform(np.array(hybrid_predictions).reshape(-1, len(required_features)))[:, 0]
    return custom_dates, hybrid_predictions

# Save prediction image
def save_prediction_image(predicted_dates, predicted_prices):
    if not os.path.exists(app.config['PREDICTIONS_FOLDER']):
        os.makedirs(app.config['PREDICTIONS_FOLDER'])

    timestamp = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    image_filename = f"prediction_{timestamp}.png"
    filepath = os.path.join(app.config['PREDICTIONS_FOLDER'], image_filename)

    plt.figure(figsize=(14, 8))
    last_days = data[-120:]
    last_days_actual = scaler.inverse_transform(data_scaled[-120:, :])[:, 0]
    plt.plot(last_days.index, last_days_actual, color='blue', label='Actual Prices (Last 120 Days)')
    plt.plot(predicted_dates, predicted_prices, color='red', linestyle='--', marker='o', label='Predicted Prices')

    plt.title('Netflix Stock Price Prediction', fontsize=16, weight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Stock Price (USD)', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    return image_filename

@app.route('/', methods=['GET', 'POST'])
def index():
    last_days = data[-120:]
    last_days['Price Change'] = last_days['Close'].diff().fillna(0)

    historical_data = [(date.strftime('%d-%m-%Y'), price, change) for date, price, change in zip(last_days.index, last_days['Close'], last_days['Price Change'])]

    if request.method == 'POST':
        date_input = request.form.get('dates')
        custom_dates = [date.strip() for date in date_input.split(",")]
        predicted_dates, predicted_prices = predict_custom_dates(custom_dates, required_features)

        price_changes = [0]
        for i in range(1, len(predicted_prices)):
            price_changes.append(predicted_prices[i] - predicted_prices[i - 1])

        formatted_dates = [date.strftime('%d-%m-%Y') for date in predicted_dates]
        predictions = [(date, price, change, "increased" if change > 0 else "decreased" if change < 0 else "no change") for date, price, change in zip(formatted_dates, predicted_prices, price_changes)]

        image_url = save_prediction_image(predicted_dates, predicted_prices)
        return render_template('index.html', image_url=image_url, predictions=predictions, historical_data=historical_data)

    return render_template('index.html', image_url=None, historical_data=historical_data)

@app.route('/static/predictions/<filename>')
def display_image(filename):
    return send_from_directory(app.config['PREDICTIONS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
