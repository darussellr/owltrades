# app.py

from flask import Flask, jsonify, request
from flask_cors import CORS
from stock_analysis import process_stock_data

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

@app.route('/')
def home():
    return "Welcome to the Stock Analysis API!", 200

@app.route('/api/stock-data')
def get_stock_data():
    symbol = request.args.get('symbol', 'SPY')
    result = process_stock_data(symbol)
    if result:
        # Convert DataFrame to list of dictionaries for JSON serialization
        result['data'] = result['data'].reset_index().to_dict('records')  # Now correctly calling reset_index()

        # Ensure that data is serialized properly for Flask
        result['ai_profit'] = int(result['ai_profit'])  # Convert int64 to int
        result['diamond_hands_profit'] = int(result['diamond_hands_profit'])  # Convert int64 to int

        return jsonify(result)
    return jsonify({'error': 'Failed to process stock data'}), 400


if __name__ == '__main__':
    app.run(debug=True)
