# app.py

from flask import Flask, jsonify, request
from stock_analysis import process_stock_data

app = Flask(__name__)

@app.route('/api/stock-data')
def get_stock_data():
    symbol = request.args.get('symbol', 'SPY')
    result = process_stock_data(symbol)
    if result:
        return jsonify(result)
    return jsonify({'error': 'Failed to process stock data'}), 400

if __name__ == '__main__':
    app.run(debug=True)