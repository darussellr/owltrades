import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Step 1: Fetch historical stock data for multiple stocks (daily intervals)
def fetch_data(symbol, start_date='2012-01-01', end_date='2024-10-04'):
    try:
        data = yf.download(symbol, start=start_date, end=end_date, interval='1d')
        return data
    except Exception as e:
        print(f"Failed to download data for {symbol}: {e}")
        return None

# Step 2: Calculate Moving Averages (50-period and 200-period for daily intervals)
def calculate_moving_averages(data):
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    return data

# Step 3: Calculate RSI (Relative Strength Index)
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

# Step 4: Calculate MACD (Moving Average Convergence Divergence)
def calculate_MACD(data):
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data

# Step 5: Generate buy/sell signals based on future price movements
def generate_future_signals(data, periods_forward=5):
    future_price = data['Close'].shift(-periods_forward)
    data['Future Return'] = (future_price - data['Close']) / data['Close']
    
    data['Buy Signal'] = np.where(data['Future Return'] > 0, 1, 0)
    data['Sell Signal'] = np.where(data['Future Return'] < 0, 1, 0)
    
    return data

# Step 6: Calculate AI model's profit based on buy/sell signals
def calculate_ai_model_profit(data, initial_investment=1000000):
    balance = initial_investment
    shares = 0
    
    for i in range(len(data)):
        if data['Buy Signal'].iloc[i] == 1 and shares == 0:
            # Buy shares
            shares = balance / data['Close'].iloc[i]
            balance = 0
        elif data['Sell Signal'].iloc[i] == 1 and shares > 0:
            # Sell shares
            balance = shares * data['Close'].iloc[i]
            shares = 0
    
    # If still holding shares at the end, sell them
    if shares > 0:
        balance = shares * data['Close'].iloc[-1]
    
    return balance

# Step 7: Calculate "Diamond Hands" profit (buy and hold strategy)
def calculate_diamond_hands_profit(data, initial_investment=1000000):
    if data.empty:
        return 0  # Skip empty data
    initial_price = data['Close'].iloc[0]
    final_price = data['Close'].iloc[-1]
    
    # Profit if holding the stock from beginning to end
    shares = initial_investment / initial_price
    final_balance = shares * final_price
    
    return final_balance

# Step 8: Interactive Plotly Graph with Stock Options
def plot_interactive_graph(stock_data, stock_symbol, ai_profit, diamond_hands_profit):
    fig = make_subplots(rows=2, cols=1, 
                        row_heights=[0.15, 0.85], 
                        shared_xaxes=True,
                        vertical_spacing=0.03)
    
    # Headers for AI and Diamond Hands profits
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='text', 
                             text=[f"AI Model Profit: ${ai_profit:,.2f}"],
                             textposition="top center", showlegend=False), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='text', 
                             text=[f"Diamond Hands Profit: ${diamond_hands_profit:,.2f}"],
                             textposition="bottom center", showlegend=False), row=1, col=1)
    
    # Plot Close price in the second row
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'],
                             mode='lines', name=f'{stock_symbol} Close Price'), row=2, col=1)

    # Plot Buy signals (slightly higher than the Close price)
    buy_signals = stock_data[stock_data['Buy Signal'] == 1]
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'] * 1.01,  # Offset by 1% for visibility
                             mode='markers', marker=dict(symbol='triangle-up', color='green', size=10),
                             name='Buy Signal'), row=2, col=1)

    # Plot Sell signals (slightly lower than the Close price)
    sell_signals = stock_data[stock_data['Sell Signal'] == 1]
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'] * 0.99,  # Offset by -1% for visibility
                             mode='markers', marker=dict(symbol='triangle-down', color='red', size=10),
                             name='Sell Signal'), row=2, col=1)

    # Update layout
    fig.update_layout(title=f'{stock_symbol} Buy and Sell Signals', xaxis_title='Date', yaxis_title='Price',
                      height=800, hovermode='x unified', margin=dict(t=100, l=50, r=50, b=50))

    return fig

# List of stocks to fetch
symbols = ['SPY', 'AAPL', 'GOOGL', 'META', 'NFLX', 'AMZN', 'TSLA']  # Use 'META' instead of 'FB'

# Step 9: Fetch data and process for each stock
stock_data_dict = {}
ai_profits = {}
diamond_hands_profits = {}

for symbol in symbols:
    print(f"Fetching data for {symbol}...")
    data = fetch_data(symbol, start_date='2012-01-01', end_date='2024-10-04')
    if data is not None and not data.empty:
        data = calculate_moving_averages(data)
        data = calculate_RSI(data)
        data = calculate_MACD(data)
        data = generate_future_signals(data)
        stock_data_dict[symbol] = data
        
        # Calculate profits for AI model and Diamond Hands strategy
        ai_profits[symbol] = calculate_ai_model_profit(data)
        diamond_hands_profits[symbol] = calculate_diamond_hands_profit(data)

# Step 10: Generate and display interactive plot for S&P 500
symbol_to_plot = 'SPY'
fig = plot_interactive_graph(stock_data_dict[symbol_to_plot], symbol_to_plot, 
                             ai_profits[symbol_to_plot], diamond_hands_profits[symbol_to_plot])

# Add buttons for selecting different stocks
fig.update_layout(
    updatemenus=[
        dict(
            buttons=list([
                dict(label="S&P 500", method="update", 
                     args=[{"x": [stock_data_dict['SPY'].index], 
                            "y": [stock_data_dict['SPY']['Close']]}, 
                           {"annotations": [{"text": f"AI Model Profit: ${ai_profits['SPY']:,.2f}", "showarrow": False},
                                            {"text": f"Diamond Hands Profit: ${diamond_hands_profits['SPY']:,.2f}", "showarrow": False}]}]),
                dict(label="Apple", method="update", 
                     args=[{"x": [stock_data_dict['AAPL'].index], 
                            "y": [stock_data_dict['AAPL']['Close']]}, 
                           {"annotations": [{"text": f"AI Model Profit: ${ai_profits['AAPL']:,.2f}", "showarrow": False},
                                            {"text": f"Diamond Hands Profit: ${diamond_hands_profits['AAPL']:,.2f}", "showarrow": False}]}]),
                dict(label="Google", method="update", 
                     args=[{"x": [stock_data_dict['GOOGL'].index], 
                            "y": [stock_data_dict['GOOGL']['Close']]}, 
                           {"annotations": [{"text": f"AI Model Profit: ${ai_profits['GOOGL']:,.2f}", "showarrow": False},
                                            {"text": f"Diamond Hands Profit: ${diamond_hands_profits['GOOGL']:,.2f}", "showarrow": False}]}]),
                dict(label="Meta", method="update", 
                     args=[{"x": [stock_data_dict['META'].index], 
                            "y": [stock_data_dict['META']['Close']]}, 
                           {"annotations": [{"text": f"AI Model Profit: ${ai_profits['META']:,.2f}", "showarrow": False},
                                            {"text": f"Diamond Hands Profit: ${diamond_hands_profits['META']:,.2f}", "showarrow": False}]}]),
                dict(label="Netflix", method="update", 
                     args=[{"x": [stock_data_dict['NFLX'].index], 
                            "y": [stock_data_dict['NFLX']['Close']]}, 
                           {"annotations": [{"text": f"AI Model Profit: ${ai_profits['NFLX']:,.2f}", "showarrow": False},
                                            {"text": f"Diamond Hands Profit: ${diamond_hands_profits['NFLX']:,.2f}", "showarrow": False}]}]),
                dict(label="Amazon", method="update", 
                     args=[{"x": [stock_data_dict['AMZN'].index], 
                            "y": [stock_data_dict['AMZN']['Close']]}, 
                           {"annotations": [{"text": f"AI Model Profit: ${ai_profits['AMZN']:,.2f}", "showarrow": False},
                                            {"text": f"Diamond Hands Profit: ${diamond_hands_profits['AMZN']:,.2f}", "showarrow": False}]}]),
                dict(label="Tesla", method="update", 
                     args=[{"x": [stock_data_dict['TSLA'].index], 
                            "y": [stock_data_dict['TSLA']['Close']]}, 
                           {"annotations": [{"text": f"AI Model Profit: ${ai_profits['TSLA']:,.2f}", "showarrow": False},
                                            {"text": f"Diamond Hands Profit: ${diamond_hands_profits['TSLA']:,.2f}", "showarrow": False}]}])
            ]),
            direction="down",
            showactive=True,
        )
    ]
)

fig.show()
