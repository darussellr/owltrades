# on_demand.py
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

def prepare_features(data):
    # Calculate all possible features
    data["SMA50"] = data["Close"].rolling(window=50).mean()
    data["SMA200"] = data["Close"].rolling(window=200).mean()
    
    # RSI
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD and Signal Line
    exp1 = data["Close"].ewm(span=12, adjust=False).mean()
    exp2 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = exp1 - exp2
    data["Signal Line"] = data["MACD"].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    data["20 Day MA"] = data["Close"].rolling(window=20).mean()
    data["20 Day STD"] = data["Close"].rolling(window=20).std()
    data["Upper Band"] = data["20 Day MA"] + (data["20 Day STD"] * 2)
    data["Lower Band"] = data["20 Day MA"] - (data["20 Day STD"] * 2)
    
    # Momentum
    data["Momentum"] = data["Close"] - data["Close"].shift(10)
    
    # Rate of Change
    data["ROC"] = data["Close"].pct_change(periods=10)
    
    # Lagged Close Prices
    for lag in range(1, 6):
        data[f"Close_Lag_{lag}"] = data["Close"].shift(lag)
    
    data.dropna(inplace=True)
    return data

def run_on_demand_model(symbol, features):
    # Fetch data
    data = yf.download(symbol, start='2010-01-01', end='2024-10-04', interval='1d')
    data = prepare_features(data)

    # Prepare target variable
    data['Future_Close'] = data['Close'].shift(-5)
    data['Price_Change'] = (data['Future_Close'] - data['Close']) / data['Close']
    data['Target'] = (data['Price_Change'] > 0.02).astype(int)  # Predict >2% increase
    data.dropna(inplace=True)

    # Prepare features and target
    X = data[features]
    y = data['Target']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Handle class imbalance
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

    # Train model
    model = XGBClassifier(scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }

    return result
