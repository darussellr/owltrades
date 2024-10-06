import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './StockAnalysisTool.css';

const StockAnalysisTool = () => {
  const [stockData, setStockData] = useState({});
  const [selectedStock, setSelectedStock] = useState('AAPL');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [visibleLines, setVisibleLines] = useState({
    Close: true,
    SMA50: true,
    SMA200: true,
    BB_upper: true,
    BB_lower: true
  });

  const symbols = ['SPY', 'AAPL', 'GOOGL', 'META', 'NFLX', 'AMZN', 'TSLA'];
  const lineColors = {
    Close: "#8884d8",
    SMA50: "#82ca9d",
    SMA200: "#ffc658",
    BB_upper: "#ff7300",
    BB_lower: "#ff7300"
  };

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

  const toggleLine = (lineName) => {
    setVisibleLines(prev => ({ ...prev, [lineName]: !prev[lineName] }));
  };

  const formatXAxis = (tickItem) => {
    const date = new Date(tickItem);
    return date.toLocaleDateString('en-US', { year: 'numeric', month: 'short' });
  };

  if (loading) return <div className="loading">Loading...</div>;
  if (error) return <div className="error">{error}</div>;

  const aiProfit = stockData.ai_model_profit || 0;
  const diamondHandsProfit = stockData.diamond_hands_profit || 0;
  const profitDifference = aiProfit - diamondHandsProfit;

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
        <div className="profit-comparison-container">
          <h2>Profit Comparison</h2>
          <div className="profit-cards">
            <div className="card ai-profit">
              <h3>AI Model Profit</h3>
              <p className="profit">{formatCurrency(aiProfit)}</p>
            </div>
            <div className="card diamond-hands-profit">
              <h3>Diamond Hands Profit</h3>
              <p className="profit">{formatCurrency(diamondHandsProfit)}</p>
            </div>
          </div>
          <div className="profit-difference">
            <h3>AI Model Outperformance</h3>
            <p className={`difference ${profitDifference >= 0 ? 'positive' : 'negative'}`}>
              {formatCurrency(Math.abs(profitDifference))}
              <span className="difference-text">
                {profitDifference >= 0 ? ' more profit' : ' less profit'}
              </span>
            </p>
          </div>
        </div>

        <div className="stats-container">
          <div className="stats-card">
            <h3>Latest Close</h3>
            <p className="stat">{formatCurrency(stockData.latest_close)}</p>
          </div>
          <div className="stats-card">
            <h3>52 Week High</h3>
            <p className="stat">{formatCurrency(stockData.fifty_two_week_high)}</p>
          </div>
          <div className="stats-card">
            <h3>52 Week Low</h3>
            <p className="stat">{formatCurrency(stockData.fifty_two_week_low)}</p>
          </div>
          <div className="stats-card">
            <h3>Total Return</h3>
            <p className="stat">{formatPercentage(stockData.total_return)}</p>
          </div>
        </div>

        <div className="chart-container">
          <h2>{selectedStock} Stock Price</h2>
          <div className="chart-toggles">
            {Object.keys(visibleLines).map((lineName) => (
              <button
                key={lineName}
                onClick={() => toggleLine(lineName)}
                className={`toggle-button ${visibleLines[lineName] ? 'active' : ''}`}
              >
                {lineName}
              </button>
            ))}
          </div>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={stockData.data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="Date" tickFormatter={formatXAxis} />
              <YAxis domain={['auto', 'auto']} />
              <Tooltip />
              <Legend />
              {Object.entries(visibleLines).map(([lineName, isVisible]) => (
                isVisible && (
                  <Line
                    key={lineName}
                    type="monotone"
                    dataKey={lineName}
                    stroke={lineColors[lineName]}
                    dot={false}
                  />
                )
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default StockAnalysisTool;