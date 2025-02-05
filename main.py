import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping
import requests


def fetch_stock_data(symbol, api_key):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    if 'Time Series (Daily)' in data:
        df = pd.DataFrame(data['Time Series (Daily)']).T.astype(float)
        df = df.iloc[::-1]
        df.index = pd.to_datetime(df.index)
        return df
    else:
        print("Data fetch error")
        return pd.DataFrame()


def prepare_data(data):
    features = pd.DataFrame({
        'open': data['1. open'],
        'high': data['2. high'],
        'low': data['3. low'],
        'close': data['4. close'],
        'volume': data['5. volume']
    })

    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)

    X_train, y_train = [], []
    for i in range(60, len(features_scaled)):
        X_train.append(features_scaled[i - 60:i])
        y_train.append(features_scaled[i, 0])

    return np.array(X_train), np.array(y_train), scaler


def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Dropout(0.4))
    model.add(Bidirectional(LSTM(50)))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def predict_future(model, X_train, scaler, future_days=60):
    new_predictions = []
    last_60_days = X_train[-1]

    for _ in range(future_days):
        x_input = last_60_days.reshape(1, 60, 5)
        next_predicted = model.predict(x_input)[0]
        new_predictions.append(next_predicted[0])
        next_predicted_full = np.full((1, 5), next_predicted[0])
        last_60_days = np.vstack([last_60_days[1:], next_predicted_full])

    new_predictions_scaled = scaler.inverse_transform(
        np.column_stack((new_predictions, np.zeros((len(new_predictions), 4))))
    )[:, 0]
    return new_predictions_scaled


def plot_predictions(data, predicted_dates, new_predictions_scaled, stock_symbol):
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['1. open'], color='black', label=f'{stock_symbol} Actual Stock Price')
    plt.plot(predicted_dates, new_predictions_scaled, color='green',
             label=f'Predicted Future {stock_symbol} Stock Price')
    plt.title(f'Extended {stock_symbol} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel(f'{stock_symbol} Stock Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    api_key = 'M77VTG3GQTMNHUKV'  # Replace with your API key
    stock_symbol = input("Enter the stock symbol: ").strip().upper()

    data = fetch_stock_data(stock_symbol, api_key)
    if data.empty:
        print("No data found. Exiting.")
        return

    X_train, y_train, scaler = prepare_data(data)

    model = build_model(input_shape=(X_train.shape[1], 5))
    model.fit(X_train, y_train, epochs=150, batch_size=32, validation_split=0.2,
              callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

    future_days = 60
    new_predictions_scaled = predict_future(model, X_train, scaler, future_days)

    predicted_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days)

    plot_predictions(data, predicted_dates, new_predictions_scaled, stock_symbol)


if __name__ == "__main__":
    main()
