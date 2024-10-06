import React, { useState, useEffect } from 'react';
import StockChart from './StockChart';
import './StockAnalysisTool.css';

const StockAnalysisTool = () => {
  const [stockData, setStockData] = useState({});
  const [selectedStock, setSelectedStock] = useState('AAPL');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const symbols = ['SPY', 'AAPL', 'GOOGL', 'META', 'NFLX', 'AMZN', 'TSLA'];

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await fetch(`http://localhost:5000/api/stock-data?symbol=${selectedStock}`);
        if (!response.ok) {
          throw new Error('Network response was not ok ' + response.statusText);
        }
        const data = await response.json();
        setStockData(data);
        setLoading(false);
      } catch (err) {
        setError('Failed to fetch stock data: ' + err.message);
        setLoading(false);
      }
    };

    fetchData();
  }, [selectedStock]);

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);
  };

  const formatPercentage = (value) => {
    return new Intl.NumberFormat('en-US', { style: 'percent', minimumFractionDigits: 2 }).format(value);
  };

  if (loading) {
    return <div className="loading">Loading...</div>;
  }

  if (error) {
    return <div className="error">{error}</div>;
  }

  return (
    <div className="container">
      <header className="header">
        <h1 className="title">Stock Analysis Tool</h1>
        <div className="stock-selector">
          <select onChange={(e) => setSelectedStock(e.target.value)} value={selectedStock}>
            {symbols.map((symbol) => (
              <option key={symbol} value={symbol}>{symbol}</option>
            ))}
          </select>
        </div>
      </header>

      <div className="dashboard">
        <div className="stats-cards">
          <div className="card">
            <h2>Latest Close</h2>
            <p className="stat">{formatCurrency(stockData.latest_close)}</p>
          </div>
          <div className="card">
            <h2>52 Week High</h2>
            <p className="stat">{formatCurrency(stockData.fifty_two_week_high)}</p>
          </div>
          <div className="card">
            <h2>52 Week Low</h2>
            <p className="stat">{formatCurrency(stockData.fifty_two_week_low)}</p>
          </div>
          <div className="card">
            <h2>Total Return</h2>
            <p className="stat">{formatPercentage(stockData.total_return)}</p>
          </div>
          <div className="card">
            <h2>AI Model Profit</h2>
            <p className="stat">{formatCurrency(stockData.ai_model_profit)}</p>
          </div>
          <div className="card">
            <h2>Diamond Hands Profit</h2>
            <p className="stat">{formatCurrency(stockData.diamond_hands_profit)}</p>
          </div>
        </div>

        <StockChart data={stockData.data} symbol={selectedStock} />
      </div>
    </div>
  );
};

export default StockAnalysisTool;   