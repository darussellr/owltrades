# stock_analysis.py
import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(symbol, start_date='2012-01-01', end_date='2024-10-04'):
    try:
        data = yf.download(symbol, start=start_date, end=end_date, interval='1d')
        return data
    except Exception as e:
        print(f"Failed to download data for {symbol}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure

def calculate_moving_averages(data):
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    return data

def calculate_RSI(data, window=14):
    # Ensure delta remains a Series
    delta = data['Close'].diff(1)
    
    # Convert delta calculations directly into Series to maintain the correct type
    gain = (delta > 0) * delta
    loss = (delta < 0) * -delta

    # Use rolling directly on Series
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    # Calculation of RSI using avg_gain and avg_loss Series
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi
    return data

def calculate_MACD(data):
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data

def generate_future_signals(data, periods_forward=5):
    future_price = data['Close'].shift(-periods_forward)
    data['Future Return'] = (future_price - data['Close']) / data['Close']
    data['Buy Signal'] = np.where(data['Future Return'] > 0, 1, 0)
    data['Sell Signal'] = np.where(data['Future Return'] < 0, 1, 0)
    return data

def process_stock_data(symbol, start_date='2012-01-01', end_date='2024-10-04'):
    data = fetch_data(symbol, start_date, end_date)
    if data.empty:
        return None  # Handle empty data gracefully

    data = calculate_moving_averages(data)
    data = calculate_RSI(data)
    data = calculate_MACD(data)
    data = generate_future_signals(data)

    # Replace NaN with None before serialization
    data.replace({np.nan: None}, inplace=True)  # More explicit handling

    # Prepare results
    ai_profit = int(data['Buy Signal'].sum())  # Convert to standard Python int
    diamond_hands_profit = int(data['Sell Signal'].sum())  # Convert to standard Python int

    # Convert date index to string for JSON serialization
    data.index = data.index.strftime('%Y-%m-%d')

    return {
        'data': data,  # Keep it as DataFrame here
        'ai_profit': ai_profit,
        'diamond_hands_profit': diamond_hands_profit
    }
