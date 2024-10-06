# pattern_recognition.py

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from ta import add_all_ta_features
from ta.utils import dropna
import json

def fetch_stock_data(symbol, start_date='2010-01-01', end_date=None):
    data = yf.download(symbol, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

def calculate_indicators(data):
    # Clean NaN values
    data = dropna(data)
    
    # Add all technical analysis features
    data = add_all_ta_features(
        data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
    )
    
    # Get the list of all columns after adding TA features
    all_columns = data.columns.tolist()
    
    # Define the desired indicators
    desired_indicators = [
        'volume_adi',
        'volume_obv',
        'volume_cmf',
        'volume_fi',
        'momentum_rsi',
        'momentum_mfi',
        'momentum_tsi',
        'momentum_uo',
        'momentum_stoch',
        'trend_macd',
        'trend_macd_signal',
        'trend_macd_diff',
        'trend_ema_fast',
        'trend_ema_slow',
        'trend_adx',
        'trend_vortex_ind_pos',
        'trend_vortex_ind_neg',
        'trend_vortex_diff',
        'trend_trix',
        'trend_mass_index',
        'trend_cci',
        'trend_dpo',
        'trend_kst',
        'trend_kst_sig',
        'trend_kst_diff',
        'trend_ichimoku_a',
        'trend_ichimoku_b',
        'trend_visual_ichimoku_a',
        'trend_visual_ichimoku_b',
        'trend_aroon_up',
        'trend_aroon_down',
        'volatility_bbm',
        'volatility_bbh',
        'volatility_bbl',
        'volatility_bbw',
        'volatility_bbp',
        'volatility_kcc',
        'volatility_kch',
        'volatility_kcl',
        'volatility_kcw',
        'volatility_kcp',
        'others_dr',
        'others_dlr',
        'others_cr',
    ]
    
    # Filter indicators to those that are actually present in the DataFrame
    indicators_to_keep = [indicator for indicator in desired_indicators if indicator in all_columns]
    
    # Include essential columns
    essential_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trend']
    
    # Select the columns
    data = data[indicators_to_keep + essential_columns if 'Trend' in data.columns else indicators_to_keep + essential_columns[:-1]]
    
    data.dropna(inplace=True)
    return data

def detect_peaks_troughs(data):
    from scipy.signal import find_peaks

    # Increase 'distance' to reduce the number of peaks and troughs
    # Peaks (Maxima)
    peaks, _ = find_peaks(data['Close'], distance=50, prominence=1)
    # Troughs (Minima)
    troughs, _ = find_peaks(-data['Close'], distance=50, prominence=1)
    data['Peak'] = data.index.isin(peaks)
    data['Trough'] = data.index.isin(troughs)
    return data, peaks, troughs

def calculate_trend_lines(data):
    # Using linear regression on the entire dataset
    X = np.arange(len(data)).reshape(-1, 1)
    y = data['Close'].values
    model = LinearRegression()
    model.fit(X, y)
    trend = model.predict(X)
    data['Trend'] = trend
    return data, model.coef_[0], model.intercept_

def identify_support_resistance(data, peaks, troughs):
    # Combine peak and trough prices
    levels = np.concatenate([data['Close'].iloc[peaks], data['Close'].iloc[troughs]])
    # Use clustering to identify levels
    from scipy.cluster.vq import kmeans
    levels = np.array(levels)
    levels = levels[~np.isnan(levels)]
    centroids, _ = kmeans(levels, 5)  # Adjust the number of levels as needed
    return centroids

def generate_predictions(data):
    # Example prediction logic based on RSI
    latest_rsi = data['momentum_rsi'].iloc[-1]
    if latest_rsi < 30:
        prediction = 'The stock is oversold. A potential upward movement is expected.'
    elif latest_rsi > 70:
        prediction = 'The stock is overbought. A potential downward movement is expected.'
    else:
        prediction = 'No strong overbought or oversold signals.'
    return prediction

def analyze_stock(symbol):
    data = fetch_stock_data(symbol)
    data, slope, intercept = calculate_trend_lines(data)
    data = calculate_indicators(data)
    data, peaks, troughs = detect_peaks_troughs(data)
    levels = identify_support_resistance(data, peaks, troughs)
    prediction = generate_predictions(data)
    
    # Prepare data for frontend
    data_json = data.to_dict('records')
    analysis = {
        'data': data_json,
        'peaks': peaks.tolist(),
        'troughs': troughs.tolist(),
        'trend_slope': slope,
        'trend_intercept': intercept,
        'support_resistance_levels': levels.tolist(),
        'prediction': prediction
    }
    return analysis

if __name__ == '__main__':
    symbol = 'AAPL'
    analysis = analyze_stock(symbol)
    # Save to JSON for frontend use
    with open(f'{symbol}_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=4)
