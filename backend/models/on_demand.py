import streamlit as st
import yfinance as yf
from joblib import dump
from xgboost import XGBClassifier
import pandas as pd
from pymongo import MongoClient
from bson.binary import Binary
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to fetch stock data using Yahoo Finance
def fetch_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date, interval='1d')
    return data

# Function to calculate features for model training
def prepare_all_features(data, symbol, start_date):
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    delta = data['Close'].diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(window=14).mean() / loss.rolling(window=14).mean()
    data['RSI'] = 100 - (100 / (1 + rs))
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['Buy Signal'] = ((data['RSI'] < 30) & (data['MACD'] > data['Signal Line'])).astype(int)
    data.dropna(inplace=True)

# Train the model with selected features
def train_xgboost_model(data, selected_features):
    X = data[selected_features].dropna()
    y = data['Buy Signal']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return model, accuracy, precision, recall, f1

# Save model to MongoDB
def save_model_to_mongo(symbol, model, accuracy, precision, recall, f1):
    client = MongoClient("mongodb+srv://ramdhanrussell:ubPAIkHJ5IKWTdox@owltrade.eh1il.mongodb.net/")
    db = client['stock_trading_models']
    
    model_data = pickle.dumps(model)
    binary_model = Binary(model_data)
    
    db.models.insert_one({
        'symbol': symbol,
        'model': binary_model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'date_saved': pd.Timestamp.now()
    })

# Streamlit UI
st.title("Stock Trading Model Generator")

# Symbol input
symbol = st.text_input("Enter the stock symbol:", value="SPY")

# Date input
start_date = st.date_input("Start date", pd.to_datetime('2012-01-01'))
end_date = st.date_input("End date", pd.to_datetime('2024-10-04'))

# Feature selection
features = ['SMA50', 'SMA200', 'RSI', 'MACD', 'Signal Line']
selected_features = [st.checkbox(feature, value=True) for feature in features]

if st.button('Train Model'):
    data = fetch_data(symbol, start_date, end_date)
    
    if data.empty:
        st.error("No data found for the selected time range.")
    else:
        prepare_all_features(data, symbol, start_date)
        model, accuracy, precision, recall, f1 = train_xgboost_model(data, selected_features)
        save_model_to_mongo(symbol, model, accuracy, precision, recall, f1)
        st.success(f"Model trained and saved to MongoDB for {symbol}.")
