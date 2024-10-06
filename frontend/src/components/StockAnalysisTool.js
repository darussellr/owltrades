import React, { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Scatter,
  ReferenceLine,
  Label
} from 'recharts';
import './StockAnalysisTool.css';

const StockAnalysisTool = () => {
  const [stockData, setStockData] = useState([]);
  const [peaksData, setPeaksData] = useState([]);
  const [troughsData, setTroughsData] = useState([]);
  const [supportResistanceLevels, setSupportResistanceLevels] = useState([]);
  const [prediction, setPrediction] = useState('');
  const [selectedStock, setSelectedStock] = useState('AAPL');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [visibleElements, setVisibleElements] = useState({
    trendLine: true,
    peaks: true,
    troughs: true,
    supportResistance: true,
  });

  const symbols = ['SPY', 'AAPL', 'GOOGL', 'META', 'NFLX', 'AMZN', 'TSLA'];

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await fetch(
            `http://localhost:5000/api/stock-analysis?symbol=${selectedStock}`
        );
        if (!response.ok) {
          throw new Error(
            'Network response was not ok ' + response.statusText
          );
        }
        const analysis = await response.json();
        setStockData(analysis.data);

        // Prepare peaks and troughs data for Scatter plots
        const peaksData = analysis.peaks.map((index) => ({
          Date: analysis.data[index].Date,
          Close: analysis.data[index].Close,
        }));
        setPeaksData(peaksData);

        const troughsData = analysis.troughs.map((index) => ({
          Date: analysis.data[index].Date,
          Close: analysis.data[index].Close,
        }));
        setTroughsData(troughsData);

        setSupportResistanceLevels(analysis.support_resistance_levels);
        setPrediction(analysis.prediction);
        setLoading(false);
      } catch (err) {
        setError('Failed to fetch stock data: ' + err.message);
        setLoading(false);
      }
    };

    fetchData();
  }, [selectedStock]);

  const formatXAxis = (tickItem) => {
    const date = new Date(tickItem);
    return date.toLocaleDateString('en-US', { year: 'numeric', month: 'short' });
  };

  const toggleElement = (element) => {
    setVisibleElements((prev) => ({
      ...prev,
      [element]: !prev[element],
    }));
  };

  if (loading) return <div className="loading">Loading...</div>;
  if (error) return <div className="error">{error}</div>;

  return (
    <div className="container">
      <header className="header">
        <h1 className="title">Stock Analysis Tool</h1>
        <div className="stock-selector">
          <select
            onChange={(e) => setSelectedStock(e.target.value)}
            value={selectedStock}
          >
            {symbols.map((symbol) => (
              <option key={symbol} value={symbol}>
                {symbol}
              </option>
            ))}
          </select>
        </div>
      </header>

      <div className="dashboard">
        <div className="chart-container">
          <h2>{selectedStock} Stock Price with Pattern Analysis</h2>
          <div className="chart-controls">
            <div className="chart-toggles">
              <button
                className={`toggle-button ${
                  visibleElements.trendLine ? 'active' : ''
                }`}
                onClick={() => toggleElement('trendLine')}
              >
                Trend Line
              </button>
              <button
                className={`toggle-button ${
                  visibleElements.peaks ? 'active' : ''
                }`}
                onClick={() => toggleElement('peaks')}
              >
                Peaks
              </button>
              <button
                className={`toggle-button ${
                  visibleElements.troughs ? 'active' : ''
                }`}
                onClick={() => toggleElement('troughs')}
              >
                Troughs
              </button>
              <button
                className={`toggle-button ${
                  visibleElements.supportResistance ? 'active' : ''
                }`}
                onClick={() => toggleElement('supportResistance')}
              >
                Support/Resistance
              </button>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={stockData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="Date" tickFormatter={formatXAxis} />
              <YAxis domain={['auto', 'auto']} />
              <Tooltip />
              <Legend />

              {/* Closing Price */}
              <Line
                type="monotone"
                dataKey="Close"
                stroke="#8884d8"
                dot={false}
                name="Closing Price"
              />

              {/* Trend Line */}
              {visibleElements.trendLine && (
                <Line
                  type="monotone"
                  dataKey="Trend"
                  stroke="#FF0000"
                  dot={false}
                  name="Trend Line"
                />
              )}

              {/* Peaks */}
              {visibleElements.peaks && (
                <Scatter
                  name="Peaks"
                  data={peaksData}
                  fill="green"
                  line={{ stroke: 'green', strokeDasharray: '3 3' }}
                />
              )}

              {/* Troughs */}
              {visibleElements.troughs && (
                <Scatter
                  name="Troughs"
                  data={troughsData}
                  fill="red"
                  line={{ stroke: 'red', strokeDasharray: '3 3' }}
                />
              )}

              {/* Support and Resistance Levels */}
              {visibleElements.supportResistance &&
                supportResistanceLevels.map((level, idx) => (
                  <ReferenceLine
                    key={`level-${idx}`}
                    y={level}
                    stroke="#000"
                    strokeDasharray="3 3"
                  >
                    <Label value={`Level ${idx + 1}`} position="insideRight" />
                  </ReferenceLine>
                ))}
            </LineChart>
          </ResponsiveContainer>

          {/* Prediction */}
          <div className="prediction-container">
            <h3>Market Prediction:</h3>
            <p>{prediction}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StockAnalysisTool;
