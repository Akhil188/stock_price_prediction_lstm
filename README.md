# Stock Market Price Prediction Using LSTM

This project implements a stock market price prediction model using Long Short-Term Memory (LSTM) networks. The model predicts future stock prices based on historical stock data. Additionally, it incorporates sentiment analysis of news headlines to factor in external market influences.

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Installation Instructions](#installation-instructions)
- [Usage Instructions](#usage-instructions)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Project Overview

The goal of this project is to predict stock market prices using historical stock data and external factors such as news headlines. The LSTM model is used for time series forecasting, and sentiment analysis is performed on the news data to enhance the predictions.

### Key Features

- Time series forecasting using LSTM
- Sentiment analysis using `TextBlob` to analyze the impact of news on stock prices
- Visualization of stock price predictions compared to actual prices

## Technologies Used

- **Python**
- **TensorFlow/Keras** - LSTM Model
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Matplotlib** - Data visualization
- **TextBlob** - Sentiment analysis

## Data Sources

- **Stock Data**: Historical stock prices can be sourced from [Yahoo Finance](https://finance.yahoo.com/) or any stock market API.
- **News Data**: Daily news headlines related to the stock market.

## Project Structure

# stock_price_prediction_lstm

stock_price_prediction/
│
├── data/
│   └── stock_data.csv             # Historical stock prices (You can download from Yahoo Finance or any API)
│   └── news_data.csv              # News headlines or articles (For sentiment analysis)
│
├── models/
│   └── lstm_model.h5              # Trained LSTM model
│
├── src/
│   ├── data_preprocessing.py      # Data loading and preprocessing functions
│   ├── lstm_model.py              # LSTM model creation and training
│   ├── sentiment_analysis.py      # Sentiment analysis on news data
│   ├── stock_price_prediction.py  # Main script for training and predicting stock prices
│
├── requirements.txt               # Dependencies
└── README.md                      # Project description


## Installation Instructions

cd stock_price_prediction_lstm
Create a virtual environment and activate it:

python -m venv venv
source venv/bin/activate         # On macOS/Linux
venv\Scripts\activate            # On Windows

Install the required dependencies:

pip install -r requirements.txt

## Usage Instructions
Add your own stock data (stock_data.csv) and news data (news_data.csv) to the data/ directory.
Run the main script to train the model and make predictions:

python src/stock_price_prediction.py
