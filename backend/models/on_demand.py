import streamlit as st
import yfinance as yf
from joblib import dump
from xgboost import XGBClassifier
import pandas as pd

# Predefined features
FEATURE_LIST = [
    'SMA50', 'SMA200', 'EMA50', 'EMA200', 'RSI', 'MACD', 'Signal Line',
    'Middle Band', 'Upper Band', 'Lower Band', 'ATR', 'Momentum',
    'Stochastic Oscillator', 'Williams %R', 'OBV', 'ROC', 'Volatility', 'Bullish_Bearish'
]

# Function to fetch stock data using Yahoo Finance
def fetch_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date, interval='1d')
    return data

# Function to calculate bullish or bearish based on the previous month
def calculate_bullish_bearish(data, symbol, start_date):
    # Fetch data from the previous month before the start date
    prev_month_start = start_date - pd.DateOffset(months=1)
    prev_month_end = start_date - pd.DateOffset(days=1)
    
    # Get data for the previous month
    prev_data = fetch_data(symbol, prev_month_start, prev_month_end)
    
    if not prev_data.empty:
        # Calculate the average closing price of the previous month
        prev_month_avg_close = prev_data['Close'].mean()
        
        # Get the closing price at the start of the selected period
        current_start_price = data['Close'].iloc[0]
        
        # Classify the period as bullish (1) if the current price is higher than the previous month’s average, bearish (-1) otherwise
        data['Bullish_Bearish'] = 1 if current_start_price > prev_month_avg_close else -1
    else:
        data['Bullish_Bearish'] = 0  # If no previous data is found, mark as neutral (could also handle this differently if needed)

# Function to calculate all features
def prepare_all_features(data, symbol, start_date):
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
    data['OBV'] = data['Volume'] * data['Direction'].cumsum()
    data['ROC'] = ((data['Close'] - data['Close'].shift(12)) / data['Close'].shift(12)) * 100
    data['Volatility'] = data['Close'].rolling(window=21).std()
    
    # Add Bullish/Bearish feature
    calculate_bullish_bearish(data, symbol, start_date)

# Function to train a model with selected features using XGBoost
def train_xgboost_model(data, selected_features):
    X = data[selected_features].dropna()
    y = data['Buy Signal']  # Target label (you may need to adjust this depending on your data)
    model = XGBClassifier(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)
    return model

# Streamlit UI
st.title("Stock Trading Model Generator")

# Symbol input
symbol = st.text_input("Enter the stock symbol (e.g., 'AAPL', 'SPY'):", value="SPY")

# Date input fields
start_date = st.date_input("Start date", pd.to_datetime('2012-01-01'))
end_date = st.date_input("End date", pd.to_datetime('2024-10-04'))

# Select features
st.write("Select the features you want to include in the model:")
selected_features = []
for feature in FEATURE_LIST:
    if st.checkbox(feature):
        selected_features.append(feature)

# Submit button to fetch data, prepare features, and train the model
if st.button('Train Model'):
    if selected_features:
        # Fetch stock data from Yahoo Finance
        data = fetch_data(symbol, start_date, end_date)
        
        if data.empty:
            st.error("No data found for the selected time frame and symbol.")
        else:
            # Prepare features
            prepare_all_features(data, symbol, start_date)
            
            # Train XGBoost model with selected features
            model = train_xgboost_model(data, selected_features)
            
            # Save the model
            model_save_path = f'models/{symbol}_custom_xgboost_model_{"_".join(selected_features)}.joblib'
            dump(model, model_save_path)
            
            st.success(f"Model trained and saved as: {model_save_path}")
    else:
        st.error("Please select at least one feature.")
