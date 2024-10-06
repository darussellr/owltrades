import yfinance as yf
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import pymongo
import pickle
from datetime import datetime
import os

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

    # Buy Signal: Next day's closing price is higher than today's
    data["Buy Signal"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    data.dropna(inplace=True)

# Train model using XGBoost
def train_model(data, features):
    X = data[features]
    y = data["Buy Signal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Check if y_train and y_test contain both classes
    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        print("Not enough classes in y_train or y_test. Skipping this period.")
        return None, None, None, None, None, None, None

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Compute confusion matrix with predefined labels
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # Compute precision, recall, f1_score
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    hyperparameters = model.get_params()

    return model, accuracy, precision, recall, f1, cm, hyperparameters

# Save the model and performance metrics to MongoDB
def save_model_to_mongo(
    symbol,
    label,
    model,
    accuracy,
    precision,
    recall,
    f1,
    cm,
    hyperparameters,
    features_used,
    trainingDataStartDate,
    trainingDataEndDate,
    db,
):
    model_data = pickle.dumps(model)

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
        "model": model_data,  # Serialized model data
        "version": 1,  # Increment if needed
        "trainedAt": datetime.utcnow(),
        "trainingDataStartDate": trainingDataStartDate,
        "trainingDataEndDate": trainingDataEndDate,
        "featuresUsed": features_used,
        "hyperparameters": hyperparameters,
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
        metrics_list = all_metrics[symbol]
        df = pd.DataFrame(metrics_list)
        avg_metrics = df.mean().to_dict()
        summary[symbol] = avg_metrics
        print(f"\nSummary for {symbol}:")
        print(f"Average Accuracy: {avg_metrics['accuracy']:.4f}")
        print(f"Average Precision: {avg_metrics['precision']:.4f}")
        print(f"Average Recall: {avg_metrics['recall']:.4f}")
        print(f"Average F1 Score: {avg_metrics['f1_score']:.4f}")

    # Overall summary
    all_metrics_combined = []
    for metrics_list in all_metrics.values():
        all_metrics_combined.extend(metrics_list)
    df_overall = pd.DataFrame(all_metrics_combined)
    avg_overall_metrics = df_overall.mean().to_dict()
    print("\nOverall Summary:")
    print(f"Average Accuracy: {avg_overall_metrics['accuracy']:.4f}")
    print(f"Average Precision: {avg_overall_metrics['precision']:.4f}")
    print(f"Average Recall: {avg_overall_metrics['recall']:.4f}")
    print(f"Average F1 Score: {avg_overall_metrics['f1_score']:.4f}")

# Train models and save to MongoDB
def create_predefined_models(symbols):
    db = connect_to_mongo()
    all_metrics = {}  # For summarizing performance metrics

    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        full_data = fetch_data(symbol, start_date="2012-01-01", end_date="2024-10-04")
        prepare_all_features(full_data)

        periods = [
            ("6m", 6),
            ("1y", 12),
            ("7y_part1", 84),
            ("7y_part2", 168),
            ("full", None),
        ]
        features = [
            "SMA50",
            "SMA200",
            "RSI",
            "MACD",
            "Signal Line",
            "Upper Band",
            "Lower Band",
            "Close",
        ]

        symbol_metrics = []  # Store metrics for this symbol

        for label, months in periods:
            period_data = full_data.tail(months * 21) if months else full_data

            if period_data.shape[0] < 50:
                print(f"Not enough data for {label} period of {symbol}. Skipping...")
                continue

            model, accuracy, precision, recall, f1, cm, hyperparameters = train_model(
                period_data, features
            )
            if model is None:
                print(f"Skipping {symbol} ({label}) due to insufficient class labels.")
                continue

            trainingDataStartDate = period_data.index.min().strftime("%Y-%m-%d")
            trainingDataEndDate = period_data.index.max().strftime("%Y-%m-%d")

            performance_metrics = save_model_to_mongo(
                symbol,
                label,
                model,
                accuracy,
                precision,
                recall,
                f1,
                cm,
                hyperparameters,
                features_used=features,
                trainingDataStartDate=trainingDataStartDate,
                trainingDataEndDate=trainingDataEndDate,
                db=db,
            )

            # Collect metrics for summarization
            symbol_metrics.append(performance_metrics)

        if symbol_metrics:
            all_metrics[symbol] = symbol_metrics

    # Summarize performance metrics
    if all_metrics:
        summarize_performance(all_metrics)
    else:
        print("No performance metrics to summarize.")

if __name__ == "__main__":
    symbols = ["SPY", "AAPL", "GOOGL", "AMZN", "TSLA", "META", "NFLX"]
    create_predefined_models(symbols)
