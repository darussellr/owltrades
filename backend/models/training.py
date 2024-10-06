import yfinance as yf
import pandas as pd
from xgboost import XGBClassifier
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pymongo
from bson.binary import Binary
import pickle

# MongoDB setup
def connect_to_mongo():
    mongo_uri = "mongodb+srv://ramdhanrussell:ubPAIkHJ5IKWTdox@owltrade.eh1il.mongodb.net/"
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
    data['RSI'] = 100 - (100 / (1 + (data['Close'].diff(1).clip(lower=0).rolling(window=14).mean() /
                                     (-data['Close'].diff(1).clip(upper=0).rolling(window=14).mean()))))
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['Buy Signal'] = ((data['RSI'] < 30) & (data['MACD'] > data['Signal Line'])).astype(int)
    data.dropna(inplace=True)

# Train model using XGBoost
def train_model(data, features):
    X = data[features]
    y = data['Buy Signal']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return model, accuracy, precision, recall, f1

# Save the model and performance metrics to MongoDB
def save_model_to_mongo(symbol, label, model, accuracy, precision, recall, f1, db):
    model_data = pickle.dumps(model)
    
    db.models.insert_one({
        'symbol': symbol,
        'label': label,
        'model': model_data,  # Storing serialized model data
        'version': 1,
        'date_saved': pd.Timestamp.now(),
        'performance': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    })
    print(f"Model for {symbol} ({label}) saved to MongoDB with performance metrics.")

# Train models and save to MongoDB
def create_predefined_models(symbols):
    db = connect_to_mongo()
    
    for symbol in symbols:
        print(f"Processing {symbol}...")
        full_data = fetch_data(symbol, start_date='2012-01-01', end_date='2024-10-04')
        prepare_all_features(full_data)
        
        periods = [('6m', 6), ('1y', 12), ('7y_part1', 84), ('7y_part2', 168), ('full', None)]
        features = ['SMA50', 'SMA200', 'RSI', 'MACD', 'Signal Line']
        
        for label, months in periods:
            period_data = full_data.tail(months * 21) if months else full_data
            
            if period_data.shape[0] < 50:
                print(f"Not enough data for {label} period of {symbol}. Skipping...")
                continue
            
            model, accuracy, precision, recall, f1 = train_model(period_data, features)
            save_model_to_mongo(symbol, label, model, accuracy, precision, recall, f1, db)

if __name__ == "__main__":
    symbols = ['SPY', 'AAPL', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX']
    create_predefined_models(symbols)
