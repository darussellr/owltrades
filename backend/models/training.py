import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pymongo
import pickle
from datetime import datetime
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# MongoDB setup
def connect_to_mongo():
    # Load MongoDB URI securely (recommended to use environment variables)
    mongo_uri = os.getenv(
        "MONGODB_URI",
        "mongodb+srv://ramdhanrussell:ubPAIkHJ5IKWTdox@owltrade.eh1il.mongodb.net/",
    )
    client = pymongo.MongoClient(mongo_uri)
    db = client["stock_prediction"]  # Database name
    return db

# Fetch stock data
def fetch_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date, interval="1d")
    return data

# Feature calculations
def prepare_all_features(data):
    # Simple Moving Averages
    data["SMA50"] = data["Close"].rolling(window=50).mean()
    data["SMA200"] = data["Close"].rolling(window=200).mean()

    # Relative Strength Index (RSI)
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
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

    # Future Price Movement
    data["Future_Close"] = data["Close"].shift(-5)
    data["Price_Change"] = (data["Future_Close"] - data["Close"]) / data["Close"]
    data["Buy_Signal"] = (data["Price_Change"] > 0.02).astype(int)  # Predict >2% increase

    data.dropna(inplace=True)

# Prepare data for LSTM
def create_sequences(X, y, time_steps=60):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model

# Train and evaluate the model
def train_evaluate_model(X_train, y_train, X_test, y_test, class_weight):
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

    # Adjusted EarlyStopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1,
        shuffle=False,
        class_weight=class_weight
    )

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return model, history, accuracy, precision, recall, f1, cm

# Save the model and performance metrics to MongoDB
def save_model_to_mongo(
    symbol,
    label,
    model,
    scaler,
    accuracy,
    precision,
    recall,
    f1,
    cm,
    features_used,
    trainingDataStartDate,
    trainingDataEndDate,
    db,
):
    # Serialize model and scaler
    # Note: For Keras models, it's better to save using model.save() method
    model_path = f"{symbol}_{label}_model.h5"
    scaler_path = f"{symbol}_{label}_scaler.pkl"
    model.save(model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Extract confusion matrix elements safely
    tn, fp, fn, tp = cm.ravel()

    performance_metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": {
            "true_positive": int(tp),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
        },
    }

    model_document = {
        "symbol": symbol,
        "label": label,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "version": 1,
        "trainedAt": datetime.utcnow(),
        "trainingDataStartDate": trainingDataStartDate,
        "trainingDataEndDate": trainingDataEndDate,
        "featuresUsed": features_used,
        "performance": performance_metrics,
    }

    # Perform upsert: update if exists, insert if not
    result = db.models.update_one(
        {"symbol": symbol, "label": label}, {"$set": model_document}, upsert=True
    )

    if result.upserted_id:
        print(
            f"Inserted new model for {symbol} ({label}) with _id: {result.upserted_id}"
        )
    else:
        print(f"Updated existing model for {symbol} ({label})")

    return performance_metrics  # Return metrics for summarization

# Summarize performance metrics
def summarize_performance(all_metrics):
    summary = {}
    for symbol in all_metrics:
        metrics = all_metrics[symbol][0]  # Only one set of metrics per symbol
        print(f"\nPerformance for {symbol}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        summary[symbol] = metrics

# Main function to run the process
def create_lstm_models(symbols):
    db = connect_to_mongo()
    all_metrics = {}  # For summarizing performance metrics

    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        data = fetch_data(symbol, start_date="2010-01-01", end_date="2024-10-04")
        prepare_all_features(data)

        features = [
            "SMA50",
            "SMA200",
            "RSI",
            "MACD",
            "Signal Line",
            "Upper Band",
            "Lower Band",
            "Momentum",
            "ROC",
            "Close",
        ]

        X = data[features]
        y = data["Buy_Signal"]

        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

        # Create sequences
        time_steps = 60  # Use 60 days of data to predict the next movement
        X_seq, y_seq = create_sequences(X_scaled, y, time_steps)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, shuffle=False
        )

        # Check if both classes are present
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"Not enough class diversity for {symbol}. Skipping...")
            continue

        # Address class imbalance
        from collections import Counter
        counter = Counter(y_train)
        majority = max(counter.values())
        class_weight = {cls: float(majority/count) for cls, count in counter.items()}

        # Train and evaluate the model
        model, history, accuracy, precision, recall, f1, cm = train_evaluate_model(
            X_train, y_train, X_test, y_test, class_weight
        )

        trainingDataStartDate = data.index.min().strftime("%Y-%m-%d")
        trainingDataEndDate = data.index.max().strftime("%Y-%m-%d")

        performance_metrics = save_model_to_mongo(
            symbol,
            label="LSTM",
            model=model,
            scaler=scaler,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            cm=cm,
            features_used=features,
            trainingDataStartDate=trainingDataStartDate,
            trainingDataEndDate=trainingDataEndDate,
            db=db,
        )

        # Collect metrics for summarization
        if performance_metrics:
            all_metrics[symbol] = [performance_metrics]

    # Summarize performance metrics
    if all_metrics:
        summarize_performance(all_metrics)
    else:
        print("No performance metrics to summarize.")

if __name__ == "__main__":
    symbols = ["SPY", "AAPL", "GOOGL", "AMZN", "TSLA", "META", "NFLX"]
    create_lstm_models(symbols)
