import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def fetch_data(symbol, start_date='2012-01-01', end_date='2024-10-04'):
    """
    Fetch stock data using yfinance for the given symbol and date range.
    """
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
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

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

def calculate_bollinger_bands(data, window=20):
    data['BB_middle'] = data['Close'].rolling(window=window).mean()
    data['BB_upper'] = data['BB_middle'] + 2 * data['Close'].rolling(window=window).std()
    data['BB_lower'] = data['BB_middle'] - 2 * data['Close'].rolling(window=window).std()
    return data

def calculate_returns(data):
    data['Daily_Return'] = data['Close'].pct_change()
    data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod() - 1
    return data

def generate_future_signals(data, periods_forward=5):
    future_price = data['Close'].shift(-periods_forward)
    data['Future Return'] = (future_price - data['Close']) / data['Close']
    
    data['Buy Signal'] = np.where(data['Future Return'] > 0, 1, 0)
    data['Sell Signal'] = np.where(data['Future Return'] < 0, 1, 0)
    
    return data

# Random Forest Model to predict Buy/Sell signals
def train_ai_model(data):
    # Step 1: Prepare the features and target
    features = ['SMA50', 'SMA200', 'RSI', 'MACD', 'Signal Line']
    data = data.dropna()  # Remove rows with missing values

    X = data[features]
    y = data['Buy Signal']  # Using 'Buy Signal' as the target

    # Step 2: Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Step 3: Train a Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 4: Predict and evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Step 5: Use the model to predict buy/sell signals on the entire dataset
    data['AI Signal'] = model.predict(X)

    return data, model

def calculate_ai_model_profit(data, initial_investment=1000000):
    balance = initial_investment
    shares = 0
    
    for i in range(len(data)):
        if data['AI Signal'].iloc[i] == 1 and shares == 0:  # AI Buy Signal
            shares = balance / data['Close'].iloc[i]
            balance = 0
        elif data['Sell Signal'].iloc[i] == 1 and shares > 0:  # AI Sell Signal
            balance = shares * data['Close'].iloc[i]
            shares = 0
    
    if shares > 0:
        balance = shares * data['Close'].iloc[-1]
    
    return balance - initial_investment

def calculate_diamond_hands_profit(data, initial_investment=1000000):
    if data.empty:
        return 0
    initial_price = data['Close'].iloc[0]
    final_price = data['Close'].iloc[-1]
    
    shares = initial_investment / initial_price
    final_balance = shares * final_price
    
    return final_balance - initial_investment

def process_stock_data(symbol, start_date='2012-01-01', end_date='2024-10-04'):
    data = fetch_data(symbol, start_date, end_date)
    if data.empty:
        return None  # Handle empty data gracefully

    data = calculate_moving_averages(data)
    data = calculate_RSI(data)
    data = calculate_MACD(data)
    data = calculate_bollinger_bands(data)
    data = calculate_returns(data)
    data = generate_future_signals(data)

    # Train AI model and add AI signals to data
    data, _ = train_ai_model(data)

    # Check if data has enough rows for the 52-week high/low calculation
    if len(data) >= 252:
        fifty_two_week_high = data['Close'].rolling(window=252).max().iloc[-1]
        fifty_two_week_low = data['Close'].rolling(window=252).min().iloc[-1]
    else:
        fifty_two_week_high = None
        fifty_two_week_low = None

    # Additional statistics
    latest_close = data['Close'].iloc[-1] if not data.empty else None
    total_return = data['Cumulative_Return'].iloc[-1] if 'Cumulative_Return' in data.columns else None
    
    # Calculate AI model and Diamond Hands profits
    ai_model_profit = calculate_ai_model_profit(data)
    diamond_hands_profit = calculate_diamond_hands_profit(data)

    # Replace NaN with None before serialization
    data.replace({np.nan: None}, inplace=True)

    # Convert date index to string and reset index to make date a column
    data = data.reset_index()
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')

    return {
        'data': data.to_dict(orient='records'),
        'latest_close': latest_close,
        'fifty_two_week_high': fifty_two_week_high,
        'fifty_two_week_low': fifty_two_week_low,
        'total_return': total_return,
        'ai_model_profit': ai_model_profit,
        'diamond_hands_profit': diamond_hands_profit
    }
