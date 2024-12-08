# Stock Sentiment Analysis and Historical Data App

This app provides sentiment analysis of stock discussions from Reddit and shows historical stock performance data from Yahoo Finance. It uses machine learning to analyze Reddit posts, predict stock sentiment, and combine this with real-time stock data to provide a comprehensive view of stock trends.

## Features:
- **Sentiment Analysis**: Analyzes Reddit discussions for sentiment related to stocks. Positive sentiment indicates that the stock will go up, and negative sentiment indicates that the stock will go down.
- **Historical Stock Data**: Fetches historical stock price trends for the selected stock from Yahoo Finance.
- **Download Data**: Allows users to download the full sentiment analysis and historical stock data.
- **Visualization**: Visualizes sentiment distribution and stock price trends.
- **Stock Prediction**: Predicts the movement of a stock based on sentiment analysis of Reddit posts.

## Setup

To run this app locally, ensure you have the following Python packages installed:

```bash
pip install pandas streamlit textblob scikit-learn yfinance joblib requests
```

## Files and Directories:
- **models/**: Directory to save the trained machine learning model.
- **logs/**: Directory to save logs for debugging and error tracking.
- **reddit_scraper.log**: Log file to track scraping issues.
- **Stock Sentiment Model**: The machine learning model used to predict stock sentiment and movement.

## Functions

### 1. `clean_text(text)`
Cleans the input text by removing punctuation, digits, and extra spaces, and converts it to lowercase.

### 2. `fetch_yahoo_finance_tickers()`
Fetches the top 100 stock tickers from Yahoo Finance. The current list includes major Indian stocks such as TCS, INFY, RELIANCE, etc.

### 3. `scrape_reddit_data(subreddit_name, limit=100)`
Scrapes Reddit posts from the specified subreddit. The `limit` parameter determines the number of posts to fetch.

### 4. `perform_sentiment_analysis(df)`
Performs sentiment analysis on Reddit posts using TextBlob. It adds a sentiment score and a sentiment label (1 for positive, 0 for negative).

### 5. `train_and_evaluate_model(df)`
Trains a machine learning model (RandomForestClassifier) on sentiment-labeled data and evaluates it using accuracy, precision, recall, and F1 score.

### 6. `fetch_historical_data(ticker, start_date, end_date)`
Fetches historical stock data (closing prices) for the selected stock from Yahoo Finance.

### 7. `display_top_5_table(df)`
Displays the top 5 rows of the provided DataFrame.

### 8. `download_csv(df, filename="data.csv")`
Generates a download button to download the given DataFrame as a CSV file.

### 9. `predict_stock_movement(model, sentiment_score)`
Predicts the stock movement based on the average sentiment score of the Reddit posts. If the sentiment score is positive, the stock is expected to go up; if negative, it is expected to go down.

## Streamlit Interface
The interface allows you to:
- Select a stock from the dynamically fetched Yahoo Finance tickers.
- Enter a subreddit to scrape Reddit posts.
- View sentiment analysis results.
- Download sentiment analysis and historical stock data.
- Visualize stock price trends and sentiment distribution.
- Get a prediction about stock movement based on sentiment.

## Example Usage:
1. **Select a Stock**: Choose from the list of top 100 stocks.
2. **Subreddit Input**: Input a subreddit name (e.g., "IndianStockMarket") to fetch posts related to stock discussions.
3. **Analyze**: Click the "Analyze" button to fetch Reddit data, perform sentiment analysis, train the model, and visualize results.
4. **Download Data**: Download sentiment analysis and stock data for further analysis.

## Logs:
Logs are saved in the `logs/reddit_scraper.log` file to track errors during Reddit scraping.

---

To run the app, simply execute the following command:

```bash
streamlit run app.py
```

Enjoy exploring stock sentiment analysis and trends!