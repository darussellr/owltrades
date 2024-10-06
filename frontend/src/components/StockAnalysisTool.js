import React, { useState, useEffect, useRef } from 'react';
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
  
  // Add zoom-related state
  const [zoomState, setZoomState] = useState({
    left: 'dataMin',
    right: 'dataMax',
    refAreaLeft: '',
    refAreaRight: '',
    animation: true
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

  // Add zoom handling functions
  const handleMouseDown = (e) => {
    if (!e) return;
    setZoomState(prev => ({
      ...prev,
      refAreaLeft: e.activeLabel
    }));
  };

  const handleMouseMove = (e) => {
    if (!e) return;
    if (zoomState.refAreaLeft) {
      setZoomState(prev => ({
        ...prev,
        refAreaRight: e.activeLabel
      }));
    }
  };

  const handleMouseUp = () => {
    if (!zoomState.refAreaLeft || !zoomState.refAreaRight) {
      setZoomState(prev => ({
        ...prev,
        refAreaLeft: '',
        refAreaRight: ''
      }));
      return;
    }

    let left = zoomState.refAreaLeft;
    let right = zoomState.refAreaRight;

    if (left > right) {
      [left, right] = [right, left];
    }

    setZoomState({
      left,
      right,
      refAreaLeft: '',
      refAreaRight: '',
      animation: true
    });
  };

  const handleZoomOut = () => {
    setZoomState({
      left: 'dataMin',
      right: 'dataMax',
      refAreaLeft: '',
      refAreaRight: '',
      animation: true
    });
  };

  if (loading) return <div className="loading">Loading...</div>;
  if (error) return <div className="error">{error}</div>;

  const aiProfit = stockData.ai_model_profit || 0;
  const diamondHandsProfit = stockData.diamond_hands_profit || 0;
  const profitDifference = aiProfit - diamondHandsProfit;

  return (
    <div className="container">
      {/* Previous header and profit comparison sections remain the same */}
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
        {/* Previous profit comparison and stats containers remain the same */}
        
        <div className="chart-container">
          <h2>{selectedStock} Stock Price</h2>
          <div className="chart-controls">
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
            <button 
              className="zoom-button" 
              onClick={handleZoomOut}
              disabled={zoomState.left === 'dataMin' && zoomState.right === 'dataMax'}
            >
              Reset Zoom
            </button>
          </div>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart
              data={stockData.data}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="Date" 
                tickFormatter={formatXAxis}
                domain={[zoomState.left, zoomState.right]}
                type="category"
                allowDataOverflow
              />
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
                    animationDuration={500}
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