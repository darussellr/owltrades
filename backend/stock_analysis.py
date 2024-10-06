# stock_analysis.py

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def fetch_data(symbol, start_date='2012-01-01', end_date='2024-10-04'):
    try:
        data = yf.download(symbol, start=start_date, end=end_date, interval='1d')
        return data
    except Exception as e:
        print(f"Failed to download data for {symbol}: {e}")
        return None

def calculate_moving_averages(data):
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    return data

def calculate_RSI(data, window=14):
    delta = data['Close'].diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()
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

def calculate_ai_model_profit(data, initial_investment=1000000):
    balance = initial_investment
    shares = 0
    
    for i in range(len(data)):
        if data['Buy Signal'].iloc[i] == 1 and shares == 0:
            shares = balance / data['Close'].iloc[i]
            balance = 0
        elif data['Sell Signal'].iloc[i] == 1 and shares > 0:
            balance = shares * data['Close'].iloc[i]
            shares = 0
    
    if shares > 0:
        balance = shares * data['Close'].iloc[-1]
    
    return balance

def calculate_diamond_hands_profit(data, initial_investment=1000000):
    if data.empty:
        return 0
    initial_price = data['Close'].iloc[0]
    final_price = data['Close'].iloc[-1]
    
    shares = initial_investment / initial_price
    final_balance = shares * final_price
    
    return final_balance

def process_stock_data(symbol, start_date='2012-01-01', end_date='2024-10-04'):
    data = fetch_data(symbol, start_date, end_date)
    if data is not None and not data.empty:
        data = calculate_moving_averages(data)
        data = calculate_RSI(data)
        data = calculate_MACD(data)
        data = generate_future_signals(data)
        
        ai_profit = calculate_ai_model_profit(data)
        diamond_hands_profit = calculate_diamond_hands_profit(data)
        
        # Convert date index to string for JSON serialization
        data.index = data.index.strftime('%Y-%m-%d')
        
        return {
            'data': data,
            'ai_profit': ai_profit,
            'diamond_hands_profit': diamond_hands_profit
        }
    return None

# Example usage
if __name__ == "__main__":
    symbols = ['SPY', 'AAPL', 'GOOGL', 'META', 'NFLX', 'AMZN', 'TSLA']
    results = {symbol: process_stock_data(symbol) for symbol in symbols}
    print(results)