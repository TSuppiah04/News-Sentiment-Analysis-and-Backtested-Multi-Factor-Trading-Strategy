import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import quantstats as qs
from dotenv import load_dotenv

from core.finnhub_news import fetch_news
from core.sentiment_scorer import finbert_init, finbert_scorer, get_sentiment_score
from core.backtest import backtest_strategy, backtest_performance, plot_equity
from core.trading_signals import momentum_signal, volatility_signal, reversion_signal, combined_signal
from core.full_report import quantstats_performance
from core.walk_forward import walk_forward
from core.threshold_testing import threshold_testing

load_dotenv()
FINNHUB_API = os.getenv("FINNHUB_API")
TICKER = "SPY"
START_DATE = "2024-01-05"
END_DATE = "2025-04-05"
ENTRY_THRESHOLD = 0.3
EXIT_THRESHOLD = 0.15

print(f"Fetching {TICKER} price data from Yahoo Finance...")
data = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
prices = data["Close"]
if isinstance(prices, pd.DataFrame):
    prices = prices.squeeze()  # convert single-column DF to Series

print(f"Fetching news data for {TICKER} from Finnhub...")
news_df = fetch_news(TICKER, START_DATE, END_DATE)

print(news_df.columns)
print(news_df.head())

daily_sentiment = get_sentiment_score(news_df, use_finbert=True, text_column="headline")

daily_sentiment = daily_sentiment.groupby(pd.to_datetime(daily_sentiment["datetime"]).dt.normalize())["sentiment_score"].mean()

momentum_series = momentum_signal(prices, window=10)
volatility_series = volatility_signal(prices, window=10)
reversion_series = reversion_signal(prices, window=10, z_value=1.0)

combined_series = combined_signal(daily_sentiment, momentum_series, volatility_series, reversion_series)

backtest_results = backtest_strategy(prices, combined_series, entry=ENTRY_THRESHOLD, exit=EXIT_THRESHOLD, stoploss=0.1)
performance = backtest_performance(backtest_results["daily_pnls"])

walk_forward_results = walk_forward(prices, combined_series, window=60, step=20, entry=[0.2, 0.3, 0.4], exit=[0.1, 0.2], stoploss=0.1)
walk_forward_performance = backtest_performance(walk_forward_results["daily_pnls"])

plot_equity(prices, backtest_results["daily_pnls"], title=f"{TICKER} Strategy vs Benchmark")
plot_equity(prices.loc[walk_forward_results["daily_pnls"].index], walk_forward_results["daily_pnls"], title=f"{TICKER} Walk-Forward Strategy vs Benchmark")
try:
    result = quantstats_performance(backtest_results, prices)
    print(f"Function returned: {result}")
except Exception as e:
    print(f"ERROR in quantstats_performance: {e}")
    import traceback
    traceback.print_exc()

try:
    result = quantstats_performance(walk_forward_results, prices.loc[walk_forward_results['daily_pnls'].index])
    print(f"Function returned: {result}")
except Exception as e:
    print(f"ERROR in quantstats_performance: {e}")
    import traceback
    traceback.print_exc()
threshold_testing(prices, combined_series)