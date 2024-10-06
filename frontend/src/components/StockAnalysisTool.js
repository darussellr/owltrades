import React, { useState, useEffect, useRef } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ReferenceArea, ReferenceLine
} from 'recharts';
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
  const [zoomState, setZoomState] = useState(null);
  const [brushPosition, setBrushPosition] = useState({ x1: null, y1: null, x2: null, y2: null });
  const chartRef = useRef(null);

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

  const handleMouseDown = (e) => {
    if (e && e.activeLabel) {
      setBrushPosition({ x1: e.activeLabel, y1: e.activeCoordinate.y, x2: null, y2: null });
    }
  };

  const handleMouseMove = (e) => {
    if (e && e.activeLabel && brushPosition.x1) {
      setBrushPosition(prev => ({ ...prev, x2: e.activeLabel, y2: e.activeCoordinate.y }));
    }
  };

  const handleMouseUp = () => {
    if (brushPosition.x1 && brushPosition.x2) {
      const [x1, x2] = [brushPosition.x1, brushPosition.x2].sort();
      setZoomState({ x1, x2 });
    }
    setBrushPosition({ x1: null, y1: null, x2: null, y2: null });
  };

  const resetZoom = () => {
    setZoomState(null);
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
        {/* Profit comparison and stats sections remain unchanged */}
        
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
          {zoomState && (
            <button onClick={resetZoom} className="reset-zoom-button">
              Reset Zoom
            </button>
          )}
          <ResponsiveContainer width="100%" height={400}>
            <LineChart
              data={stockData.data}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              ref={chartRef}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="Date"
                tickFormatter={formatXAxis}
                domain={zoomState ? [zoomState.x1, zoomState.x2] : ['auto', 'auto']}
                type="category"
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
                  />
                )
              ))}
              {brushPosition.x1 && brushPosition.x2 && (
                <ReferenceArea
                  x1={brushPosition.x1}
                  x2={brushPosition.x2}
                  strokeOpacity={0.3}
                  fill="blue"
                  fillOpacity={0.1}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default StockAnalysisTool;