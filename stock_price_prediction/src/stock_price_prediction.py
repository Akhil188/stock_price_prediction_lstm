import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from .data_preprocessing import load_stock_data, preprocess_stock_data, create_dataset
from .lstm_model import create_lstm_model, train_lstm_model
from .sentiment_analysis import load_news_data, perform_sentiment_analysis, aggregate_sentiments

# Load data
stock_df = load_stock_data('./data/stock_data.csv')

# Preprocess data
scaled_data, scaler = preprocess_stock_data(stock_df)

# Create dataset
time_step = 60
X, y = create_dataset(scaled_data, time_step)

# Reshape data for LSTM input
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create and train LSTM model
model = create_lstm_model((X_train.shape[1], 1))
train_lstm_model(model, X_train, y_train)

# Predict stock prices
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")

# Plot predictions vs actual stock prices
plt.figure(figsize=(14, 5))
plt.plot(stock_df.index[train_size + time_step:], y_test, label="Actual Stock Price")
plt.plot(stock_df.index[train_size + time_step:], y_pred, label="Predicted Stock Price")
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Sentiment analysis on news data
news_df = load_news_data('./data/news_data.csv')
news_df = perform_sentiment_analysis(news_df)
daily_sentiments = aggregate_sentiments(news_df)

print("Sentiment analysis complete.")
# Save the trained model to the 'models' directory in the new Keras format
model.save('./models/lstm_stock_price_model.keras')
print("Model saved to './models/lstm_stock_price_model.keras'")
