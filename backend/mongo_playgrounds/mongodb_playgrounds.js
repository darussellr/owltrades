/* global use, db */
// MongoDB Playground

// Use the appropriate database
use('stock_trading_models');

// Insert a model without preset performance metrics.
// This would normally be handled by your backend or training process after training completes.
db.getCollection('models').insertOne({
    "symbol": "AAPL",  // Stock symbol
    "label": "6m",  // Model label (e.g., 6 months data, full history)
    "training_start_date": new Date('2023-04-01'),  // Start of training period
    "training_end_date": new Date('2023-10-01'),  // End of training period
    "model_hyperparameters": {  // Store hyperparameters used for training
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },
    "model": Binary(binaryModelData),  // Store the binary model file
    "version": 1,  // Model versioning for retraining purposes
    "date_saved": new Date()  // Timestamp of when the model was saved
});

// Fetch all models stored for any stock symbol (e.g., SPY)
const modelsForSPY = db.getCollection('models').find({
    symbol: "SPY"
}).toArray();
console.log("Models for SPY:", modelsForSPY);

// Fetch performance metrics for the most recent model for SPY
// These metrics would be dynamically updated after model evaluation in your backend or training process.
const recentModelPerformance = db.getCollection('models').find({
    symbol: "SPY"
}).sort({ date_saved: -1 }).limit(1).toArray();
console.log("Most recent model performance for SPY:", recentModelPerformance);

// Calculate average accuracy across all models for SPY
// Performance metrics should already be saved dynamically during model evaluation.
db.getCollection('models').aggregate([
    { $match: { symbol: "SPY" } },
    { $group: { _id: "$symbol", avg_accuracy: { $avg: "$performance.accuracy" } } }
]).toArray();

// Fetch predictions for a specific time range and stock symbol (e.g., SPY)
const recentPredictions = db.getCollection('predictions').find({
    symbol: "SPY",
    date: { $gte: new Date('2023-10-01'), $lt: new Date('2023-11-01') }
}).toArray();
console.log("Recent predictions for SPY:", recentPredictions);
