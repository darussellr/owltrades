import yfinance as yf
import pandas as pd
from xgboost import XGBClassifier
from joblib import dump, load
import os
import pymongo
from bson.binary import Binary
import pickle

# MongoDB setup
def connect_to_mongo():
    # Replace with your MongoDB Atlas connection string
    mongo_uri = "your_mongo_atlas_connection_string"
    client = pymongo.MongoClient(mongo_uri)
    db = client['stock_trading_models']  # Database name
    return db

# Fetch stock data
def fetch_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date, interval='1d')
    return data

# Feature calculations
def prepare_all_features(data):
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()
    delta = data['Close'].diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['Middle Band'] = data['Close'].rolling(window=20).mean()
    data['Upper Band'] = data['Middle Band'] + 2 * data['Close'].rolling(window=20).std()
    data['Lower Band'] = data['Middle Band'] - 2 * data['Close'].rolling(window=20).std()
    data['High-Low'] = data['High'] - data['Low']
    data['High-Close'] = abs(data['High'] - data['Close'].shift(1))
    data['Low-Close'] = abs(data['Low'] - data['Close'].shift(1))
    data['True Range'] = data[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    data['ATR'] = data['True Range'].rolling(window=14).mean()
    data['Momentum'] = data['Close'] - data['Close'].shift(10)
    data['Lowest Low'] = data['Low'].rolling(window=14).min()
    data['Highest High'] = data['High'].rolling(window=14).max()
    data['Stochastic Oscillator'] = 100 * (data['Close'] - data['Lowest Low']) / (data['Highest High'] - data['Lowest Low'])
    data['Williams %R'] = (data['Highest High'] - data['Close']) / (data['Highest High'] - data['Lowest Low']) * -100
    data['Direction'] = (data['Close'] > data['Close'].shift(1)).astype(int)
    data['OBV'] = (data['Volume'] * data['Direction']).cumsum()
    data['ROC'] = ((data['Close'] - data['Close'].shift(12)) / data['Close'].shift(12)) * 100
    data['Volatility'] = data['Close'].rolling(window=21).std()

    # Example Buy Signal: You need to define how 'Buy Signal' is determined
    data['Buy Signal'] = ((data['RSI'] < 30) & (data['MACD'] > data['Signal Line'])).astype(int)

    # Drop rows with any NaN values resulting from feature calculations
    data.dropna(inplace=True)

def train_model(data, features):
    X = data[features]
    y = data['Buy Signal']

    # Initialize XGBoost classifier
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # Fit the model
    model.fit(X, y)
    return model

# Save the model to MongoDB
def save_model_to_mongo(symbol, label, model, db):
    model_data = pickle.dumps(model)
    binary_model = Binary(model_data)
    
    # Insert model into MongoDB
    db.models.insert_one({
        'symbol': symbol,
        'label': label,
        'model': binary_model,
        'features': ['SMA50', 'SMA200', 'EMA50', 'EMA200', 'RSI', 'MACD', 'Signal Line', 
                     'Middle Band', 'Upper Band', 'Lower Band', 'ATR', 'Momentum', 
                     'Stochastic Oscillator', 'Williams %R', 'OBV', 'ROC', 'Volatility']
    })
    print(f"Model for {symbol} ({label}) saved to MongoDB.")

# Create models and store them in MongoDB for multiple stocks
def create_predefined_models(symbols):
    db = connect_to_mongo()  # Connect to MongoDB

    for symbol in symbols:
        print(f"Processing {symbol}...")
        full_data = fetch_data(symbol, start_date='2012-01-01', end_date='2024-10-04')
        prepare_all_features(full_data)
        
        # Define the periods and train models
        periods = [('6m', 6), ('1y', 12), ('7y_part1', 84), ('7y_part2', 168), ('full', None)]
        features = [
            'SMA50', 'SMA200', 'EMA50', 'EMA200', 'RSI', 'MACD', 'Signal Line',
            'Middle Band', 'Upper Band', 'Lower Band', 'ATR', 'Momentum',
            'Stochastic Oscillator', 'Williams %R', 'OBV', 'ROC', 'Volatility'
        ]
        
        for label, months in periods:
            if months:
                period_data = full_data.tail(months * 21)  # Approx 21 trading days per month
            else:
                period_data = full_data
            
            # Check if there is enough data
            if period_data.shape[0] < 50:  # Example threshold
                print(f"Not enough data for period {label} for {symbol}. Skipping...")
                continue

            # Train model using all features
            model = train_model(period_data, features)
            
            # Save the model to MongoDB
            save_model_to_mongo(symbol, label, model, db)

# Example usage: Create models for multiple stocks and store them in MongoDB
if __name__ == "__main__":
    symbols = ['SPY', 'AAPL', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX']  # List of stock symbols
    create_predefined_models(symbols)
