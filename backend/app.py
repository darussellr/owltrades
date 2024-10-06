# app.py

from flask import Flask, jsonify, request
from flask_cors import CORS
from stock_analysis import process_stock_data
from models.pattern_recognition import analyze_stock

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
        return jsonify(result)
    return jsonify({'error': 'Failed to process stock data'}), 400

@app.route('/api/stock-analysis', methods=['GET'])
def stock_analysis():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': 'Symbol parameter is required.'}), 400
    try:
        analysis = analyze_stock(symbol)
        return jsonify(analysis)
    except Exception as e:
        print(f'Error analyzing stock {symbol}: {e}')
        return jsonify({'error': 'Failed to analyze stock data.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
