import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const StockChart = ({ data, symbol }) => {
  // Function to format the date ticks
  const formatXAxis = (tickItem) => {
    const date = new Date(tickItem);
    return date.getFullYear();
  };

  // Custom tooltip to display full date and values
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const date = new Date(label);
      return (
        <div className="custom-tooltip">
          <p className="label">{`Date: ${date.toLocaleDateString()}`}</p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }}>
              {`${entry.name}: ${entry.value.toFixed(2)}`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="chart-container">
      <h2>{symbol} Stock Price</h2>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="Date" 
            tickFormatter={formatXAxis}
            interval={'preserveStartEnd'}
            minTickGap={50}
          />
          <YAxis domain={['auto', 'auto']} />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          <Line type="monotone" dataKey="Close" stroke="#007bff" dot={false} />
          <Line type="monotone" dataKey="SMA50" stroke="#28a745" dot={false} />
          <Line type="monotone" dataKey="SMA200" stroke="#ffc107" dot={false} />
          <Line type="monotone" dataKey="BB_upper" stroke="#17a2b8" dot={false} />
          <Line type="monotone" dataKey="BB_lower" stroke="#17a2b8" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default StockChart;