// MongoDB Playground

// Define the database and collection
const db = connect('mongodb+srv://ramdhanrussell:ubPAIkHJ5IKWTdox@owltrade.eh1il.mongodb.net/');

// Use the stock_trading_models database
const database = db.getSiblingDB('stock_trading_models');

// Create a sample model document to be inserted
const modelData = {
    symbol: "AAPL",                   // Stock symbol
    label: "7y_part1",                // Model label, e.g., 6m, 1y, 7y_part1, etc.
    parameters: {                     // XGBoost model parameters used for training
        n_estimators: 100,
        learning_rate: 0.1,
        max_depth: 5,
        subsample: 0.8,
        colsample_bytree: 0.8,
        eval_metric: 'logloss'
    },
    features: [                       // Features used in the model
        'SMA50', 
        'SMA200', 
        'RSI', 
        'MACD', 
        'Signal Line'
    ],
    performance: {                    // Model performance metrics
        accuracy: 0.75,               // Example accuracy
        precision: 0.72,              // Example precision
        recall: 0.68,                 // Example recall
        f1_score: 0.7                 // Example F1-score
    },
    version: 1,                       // Model version (you can increment this for new versions)
    date_saved: new Date(),           // Timestamp of model saving
    additional_info: {                // Optional: Any other relevant info
        train_test_split: 0.2,        // Train-test split ratio
        time_period: "7 years",       // Time period used for training (based on stock data)
        data_points: 1470             // Number of data points (rows) used for training
    },
    model_binary: Binary(pickle.dumps("dummy_model_data")) // This would be the actual binary of the trained model
};

// Insert the model document into the models collection
database.models.insertOne(modelData);

// Query the models collection to verify the insert
database.models.find({
    symbol: "AAPL",
    label: "7y_part1"
}).pretty();
