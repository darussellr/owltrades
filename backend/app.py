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
        # The data is already in the correct format (a list of dictionaries), so we don't need to reset the index
        return jsonify(result)
    return jsonify({'error': 'Failed to process stock data'}), 400


if __name__ == '__main__':
    app.run(debug=True)
