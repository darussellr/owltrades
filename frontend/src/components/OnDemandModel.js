// OnDemandModel.js
import React, { useState } from 'react';
import './OnDemandModel.css';

const OnDemandModel = () => {
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [selectedStock, setSelectedStock] = useState('AAPL');
  const [modelResult, setModelResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const allFeatures = [
    'SMA50',
    'SMA200',
    'RSI',
    'MACD',
    'Signal Line',
    'Upper Band',
    'Lower Band',
    'Momentum',
    'ROC',
    'Close',
    'Close_Lag_1',
    'Close_Lag_2',
    'Close_Lag_3',
    'Close_Lag_4',
    'Close_Lag_5',
  ];

  const symbols = ['SPY', 'AAPL', 'GOOGL', 'META', 'NFLX', 'AMZN', 'TSLA'];

  const handleFeatureChange = (e) => {
    const feature = e.target.value;
    setSelectedFeatures((prevFeatures) =>
      e.target.checked
        ? [...prevFeatures, feature]
        : prevFeatures.filter((f) => f !== feature)
    );
  };

  const runModel = async () => {
    if (selectedFeatures.length === 0) {
      alert('Please select at least one feature.');
      return;
    }

    setLoading(true);
    setModelResult(null);

    try {
      const response = await fetch('http://localhost:5000/api/run-on-demand-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: selectedStock,
          features: selectedFeatures,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to run the model.');
      }

      const data = await response.json();
      setModelResult(data);
    } catch (error) {
      console.error('Error:', error);
      alert('Error running the model.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="on-demand-container">
      <h2>On-Demand Model</h2>

      <div className="stock-selector">
        <label htmlFor="stock-select">Select Stock Symbol:</label>
        <select
          id="stock-select"
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

      <div className="feature-selector">
        <h3>Select Features:</h3>
        <div className="features-grid">
          {allFeatures.map((feature) => (
            <div key={feature} className="feature-option">
              <input
                type="checkbox"
                id={feature}
                value={feature}
                onChange={handleFeatureChange}
              />
              <label htmlFor={feature}>{feature}</label>
            </div>
          ))}
        </div>
      </div>

      <button className="run-model-button" onClick={runModel} disabled={loading}>
        {loading ? 'Running Model...' : 'Run Model'}
      </button>

      {modelResult && (
        <div className="model-result">
          <h3>Model Performance Metrics:</h3>
          <p>Accuracy: {modelResult.accuracy.toFixed(4)}</p>
          <p>Precision: {modelResult.precision.toFixed(4)}</p>
          <p>Recall: {modelResult.recall.toFixed(4)}</p>
          <p>F1 Score: {modelResult.f1_score.toFixed(4)}</p>
        </div>
      )}
    </div>
  );
};

export default OnDemandModel;
