import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const StockAnalysisTool = () => {
  const [stockData, setStockData] = useState({});
  const [selectedStock, setSelectedStock] = useState('SPY');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const symbols = ['SPY', 'AAPL', 'GOOGL', 'META', 'NFLX', 'AMZN', 'TSLA'];

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        // In a real application, you would fetch this data from your backend
        const response = await fetch(`http://localhost:5000/api/stock-data?symbol=${selectedStock}`);
        const data = await response.json();
        setStockData(data);
        setLoading(false);
      } catch (err) {
        setError('Failed to fetch stock data');
        setLoading(false);
      }
    };

    fetchData();
  }, [selectedStock]);

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);
  };

  if (loading) {
    return <div className="loading">Loading...</div>;
  }

  if (error) {
    return <div className="error">{error}</div>;
  }

  return (
    <div className="container">
      <h1 className="title">Stock Analysis Tool</h1>
      
      <div className="stock-selector">
        <select onChange={(e) => setSelectedStock(e.target.value)} value={selectedStock}>
          {symbols.map((symbol) => (
            <option key={symbol} value={symbol}>{symbol}</option>
          ))}
        </select>
      </div>

      <div className="profit-cards">
        <div className="card">
          <h2>AI Model Profit</h2>
          <p className="profit">{formatCurrency(stockData.ai_profit)}</p>
        </div>
        <div className="card">
          <h2>Diamond Hands Profit</h2>
          <p className="profit">{formatCurrency(stockData.diamond_hands_profit)}</p>
        </div>
      </div>

      <div className="chart-container">
        <h2>{selectedStock} Stock Price</h2>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={stockData.data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="Close" stroke="#8884d8" dot={false} />
            <Line type="monotone" dataKey="SMA50" stroke="#82ca9d" dot={false} />
            <Line type="monotone" dataKey="SMA200" stroke="#ffc658" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default StockAnalysisTool;