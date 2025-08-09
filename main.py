import os
import datetime
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from finnhub_news import fetch_news
from sentiment_scorer import finbert_init, finbert_scorer, get_sentiment_score
from backtest import backtest_strategy, backtest_performance, plot_equity
from trading_signals import momentum_signal, volatility_signal, reversion_signal, combined_signal

load_dotenv()
FINNHUB_API = os.getenv("FINNHUB_API")
TICKER = "AAPL"
START_DATE = "2023-01-01"
END_DATE = "2023-01-31"
ENTRY_THRESHOLD = 0.25

print(f"Fetching {TICKER} price data from Yahoo Finance...")
data = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
prices = data["Close"]

print(f"Fetching news data for {TICKER} from Finnhub...")
news_df = fetch_news(TICKER, START_DATE, END_DATE)
daily_sentiment = get_sentiment_score(news_df, use_finbert=True, text_column="headline")

daily_sentiment = daily_sentiment.groupby(pd.to_datetime(daily_sentiment["datetime"]).dt.normalize())["sentiment_score"].mean()

momentum_series = momentum_signal(prices, window=10)
volatility_series = volatility_signal(prices, window=10)
reversion_series = reversion_signal(prices, window=10, z_value=1.0)

combined_series = combined_signal(daily_sentiment, momentum_series, volatility_series, reversion_series)

backtest_results = backtest_strategy(prices, combined_series, entry=ENTRY_THRESHOLD, exit=0.0, stoploss=0.1)
