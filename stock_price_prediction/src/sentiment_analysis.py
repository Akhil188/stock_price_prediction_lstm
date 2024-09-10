import pandas as pd
from textblob import TextBlob

def load_news_data(file_path):
    news_df = pd.read_csv(file_path)
    return news_df

def perform_sentiment_analysis(news_df):
    news_df['Sentiment'] = news_df['Headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return news_df

def aggregate_sentiments(news_df):
    daily_sentiment = news_df.groupby('Date')['Sentiment'].mean()
    return daily_sentiment
