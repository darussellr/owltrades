// Import necessary modules
const { MongoClient } = require('mongodb');
require('dotenv').config(); // For loading environment variables

// Load environment variables from a .env file (Make sure to create this file and add it to .gitignore)
const uri = process.env.MONGODB_URI; // Your MongoDB URI stored in an environment variable

// Function to connect to MongoDB
async function connectToMongoDB() {
  const client = new MongoClient(uri, { useUnifiedTopology: true });
  try {
    // Connect to the MongoDB cluster
    await client.connect();
    console.log('Connected to MongoDB');
    return client;
  } catch (err) {
    console.error('MongoDB connection error:', err);
    throw err;
  }
}

// Function to store performance metrics
async function storePerformanceMetrics(metrics) {
  const client = await connectToMongoDB();
  try {
    const database = client.db('stock_prediction');
    const collection = database.collection('model_performance');

    // Insert the performance metrics document
    const result = await collection.insertOne(metrics);
    console.log(`Metrics inserted with _id: ${result.insertedId}`);
  } catch (err) {
    console.error('Error inserting metrics:', err);
  } finally {
    // Close the connection
    await client.close();
  }
}

// Function to update the model in real-time
async function updateModelInRealtime(newMetrics) {
  const client = await connectToMongoDB();
  try {
    const database = client.db('stock_prediction');
    const collection = database.collection('model_performance');

    // Update the existing document with new metrics
    const filter = { modelVersion: newMetrics.modelVersion };
    const options = { upsert: true };
    const updateDoc = {
      $set: newMetrics,
    };

    const result = await collection.updateOne(filter, updateDoc, options);
    if (result.upsertedCount > 0) {
      console.log(`Inserted a new document with _id: ${result.upsertedId._id}`);
    } else {
      console.log(`Updated existing document with modelVersion: ${newMetrics.modelVersion}`);
    }
  } catch (err) {
    console.error('Error updating metrics:', err);
  } finally {
    await client.close();
  }
}

// Example usage
async function main() {
  // Example performance metrics to store
  const metrics = {
    modelVersion: '1.0.0',
    trainedAt: new Date(),
    trainingDataStartDate: '2010-01-01',
    trainingDataEndDate: '2022-01-01',
    featuresUsed: [
      'SMA_5',
      'SMA_10',
      'SMA_15',
      'EMA_5',
      'EMA_10',
      'RSI',
      'MACD',
      'BB_Upper',
      'BB_Lower',
    ],
    hyperparameters: {
      n_estimators: 100,
      random_state: 42,
    },
    evaluationMetrics: {
      accuracy: 0.65,
      precision: 0.66,
      recall: 0.64,
      f1_score: 0.65,
      confusionMatrix: {
        truePositive: 500,
        trueNegative: 450,
        falsePositive: 250,
        falseNegative: 300,
      },
    },
    backtestingResults: {
      strategyReturn: 1.50,
      buyHoldReturn: 1.20,
      sharpeRatio: 1.25,
      maxDrawdown: -0.15,
    },
  };

  // Store the metrics
  await storePerformanceMetrics(metrics);

  // Suppose we have new metrics after retraining the model
  const updatedMetrics = {
    ...metrics,
    modelVersion: '1.0.1',
    trainedAt: new Date(),
    evaluationMetrics: {
      accuracy: 0.68,
      precision: 0.70,
      recall: 0.66,
      f1_score: 0.68,
      confusionMatrix: {
        truePositive: 520,
        trueNegative: 460,
        falsePositive: 230,
        falseNegative: 290,
      },
    },
    backtestingResults: {
      strategyReturn: 1.55,
      buyHoldReturn: 1.20,
      sharpeRatio: 1.30,
      maxDrawdown: -0.14,
    },
  };

  // Update the model with new metrics
  await updateModelInRealtime(updatedMetrics);
}

main().catch(console.error);
