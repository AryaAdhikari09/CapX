import pandas as pd
import streamlit as st
from textblob import TextBlob
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yfinance as yf
import joblib
import os
import logging
import requests

# Create logs directory if not exists
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(filename="logs/reddit_scraper.log", level=logging.INFO)

# Ensure the 'models' directory exists
if not os.path.exists("models"):
    os.makedirs("models")


# Function to clean text data
def clean_text(text):
    text = "".join([char for char in text if char not in string.punctuation])
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Function to fetch top 100 stocks from Yahoo Finance
def fetch_yahoo_finance_tickers():
    tickers = yf.Tickers(
        "TCS INFY RELIANCE HDFCBANK ICICIBANK SBIN HINDUNILVR LT BAJFINANCE KOTAKBANK AXISBANK ITC MARUTI ASIANPAINT HCLTECH BHARTIARTL WIPRO NTPC M&M TITAN ULTRACEMCO BAJAJ-AUTO SUNPHARMA CIPLA TECHM ULTRATECH HDFC HDFCLIFE POWERGRID BRITANNIA JSWSTEEL EICHERMOT DRREDDY TATASTEEL BAJAJFINSV ICICIGI INDUSINDBK HINDALCO SBILIFE MOTHERSUMI BHARTIARTL PVR INOXLEISURE SHREECEM ADANIGREEN ADANIPORTS HAVELLS BIOCON GODREJCP MUTHOOTFIN GAIL MINDTREE JINDALSTEL DIVISLAB CONCOR DABUR HINDPETRO MOTHERSUMI GLENMARK PELPEL VOLTAS COALINDIA TATAPOWER MARUTI MCDOWELL-N RBLBANK TATAMOTORS TATAELXSI JSWENERGY UPL PFC MGL FEDERALBNK RAYMOND PETRONET REC HDFCAMC LUPIN MOTHERSON TATACONSUMT EDELWEISS JUBLFOOD CASTROL HINDZINC IEX KPRMILL BHEL KIRLOSKAR OIL CUMMINS VSTIND HCLTECH SAIL NILKAMAL TATAGLOBAL CONCOR BELBHEL AMARAJABAT JSWSTEEL ITC HCLTECH PEL"
    )
    stock_dict = tickers.tickers
    return list(stock_dict.keys())[:100]


# Function to scrape Reddit data
def scrape_reddit_data(subreddit_name, limit=100):
    url = f"https://www.reddit.com/r/{subreddit_name}/hot.json?limit={limit}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        logging.error(
            f"Failed to retrieve data from Reddit for subreddit {subreddit_name}."
        )
        return None

    posts = response.json()["data"]["children"]
    data = []

    for post in posts:
        title = post["data"]["title"]
        score = post["data"]["score"]
        num_comments = post["data"]["num_comments"]
        selftext = post["data"]["selftext"]
        data.append([title, score, num_comments, selftext])

    df = pd.DataFrame(data, columns=["title", "score", "num_comments", "selftext"])
    return df


# Sentiment analysis
def perform_sentiment_analysis(df):
    df["content"] = df["title"] + " " + df["selftext"]
    df["sentiment"] = df["content"].apply(lambda x: TextBlob(x).sentiment.polarity)
    df["sentiment_label"] = df["sentiment"].apply(
        lambda x: 1 if x > 0 else 0
    )  # Positive = 1, Negative = 0
    return df


# Train and evaluate machine learning model
def train_and_evaluate_model(df):
    df = df[["score", "num_comments", "sentiment", "sentiment_label"]]
    X = df.drop("sentiment_label", axis=1)
    y = df["sentiment_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Ensure the 'models' directory exists before saving the model
    if not os.path.exists("models"):
        os.makedirs("models")

    # Save the model using joblib
    model_filename = "models/reddit_stock_sentiment_model.pkl"
    joblib.dump(model, model_filename)  # Save the model to disk
    logging.info(f"Model saved to {model_filename}")

    return model, accuracy, precision, recall, f1


# Function to fetch historical stock data
def fetch_historical_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data[["Close"]]


# Function to display top 5 rows of the dataframe
def display_top_5_table(df):
    st.table(df.head())  # Display top 5 rows of the dataframe


# Function to download dataframe as CSV
def download_csv(df, filename="data.csv"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download full data", data=csv, file_name=filename, mime="text/csv"
    )


# Predict stock movement based on sentiment analysis
def predict_stock_movement(model, sentiment_score):
    if sentiment_score > 0:
        return "Stock will go up"
    else:
        return "Stock will go down"


# Streamlit interface
def app():
    # Sidebar content
    st.sidebar.title("About the App")
    st.sidebar.write(
        """
        This app provides sentiment analysis of stock discussions from Reddit and shows 
        historical stock performance data from Yahoo Finance. It leverages the power of 
        machine learning to analyze Reddit posts, predicting stock sentiment, and combines 
        it with real-time stock data to give a comprehensive view of stock trends.

        You can:
        - Analyze Reddit discussions for sentiment regarding stocks
        - Visualize stock price trends over a specified date range
        - Download the full data for further analysis

        ### About Me:
        An undergraduate student with a diverse skill set across computer science. By coding, designing, and producing audio content, I integrate technical expertise with creativity. I am eager to harness and apply these skills to innovate and make a meaningful impact in the field :)"""
    )

    # Main Title
    st.title("Stock Sentiment Analysis and Historical Data")

    # Date Input Fields
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))

    # Fetch stock tickers dynamically from Yahoo Finance
    if "stock_tickers" not in st.session_state:
        st.session_state["stock_tickers"] = fetch_yahoo_finance_tickers()

    # Select a stock
    selected_stock = st.selectbox(
        "Select a Stock",
        st.session_state["stock_tickers"],
        help="Choose a stock to analyze",
    )

    # Input subreddit and number of posts
    subreddit_name = st.text_input("Enter Subreddit Name", "IndianStockMarket")
    limit = st.slider("Number of Posts to Analyze", 10, 500, 100)

    # Scrape and analyze data
    if st.button("Analyze"):
        st.write(f"Fetching data from r/{subreddit_name}...")

        # Fetch Reddit data
        df = scrape_reddit_data(subreddit_name, limit)

        if df is not None:
            # Perform sentiment analysis
            df = perform_sentiment_analysis(df)

            # Rearrange columns: move 'sentiment' and 'sentiment_label' to the front
            df = df[
                [
                    "sentiment",
                    "sentiment_label",
                    "title",
                    "score",
                    "num_comments",
                    "selftext",
                ]
            ]

            # Display the top 5 rows of the merged data
            st.subheader("Scrapped Reddit Data + Sentiment Analysis")
            display_top_5_table(df)

            # Provide download link for full sentiment analysis data
            download_csv(df, filename="reddit_stock_sentiment_data.csv")

            # Visualize sentiment distribution
            st.subheader("Sentiment Distribution")
            sentiment_count = df["sentiment_label"].value_counts()
            st.bar_chart(sentiment_count)

            # Train and evaluate the model
            st.write("Training Model...")
            model, accuracy, precision, recall, f1 = train_and_evaluate_model(df)

            # Display model evaluation results in a table format
            st.subheader(f"Model Evaluation for {selected_stock}")
            evaluation_data = {
                "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
                "Score": [accuracy, precision, recall, f1],
            }
            evaluation_df = pd.DataFrame(evaluation_data)
            st.table(evaluation_df)

            # Fetch historical stock data
            stock_data = fetch_historical_data(selected_stock, start_date, end_date)

            st.subheader(f"Stock Price Trend for {selected_stock}")
            st.line_chart(stock_data["Close"])

            st.write(f"Historical Stock Data for {selected_stock}:")
            display_top_5_table(stock_data)

            # Provide download link for historical stock data
            download_csv(stock_data, filename=f"{selected_stock}_historical_data.csv")

            # Predict stock movement based on sentiment of Reddit posts
            sentiment_score = df["sentiment"].mean()  # average sentiment score
            prediction = predict_stock_movement(model, sentiment_score)

            st.header(f"Prediction for {selected_stock}: {prediction}")


if __name__ == "__main__":
    app()
