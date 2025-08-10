import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import quantstats as qs
from dotenv import load_dotenv

from finnhub_news import fetch_news
from sentiment_scorer import finbert_init, finbert_scorer, get_sentiment_score
from backtest import backtest_strategy, backtest_performance, plot_equity
from trading_signals import momentum_signal, volatility_signal, reversion_signal, combined_signal
from full_report import quantstats_performance

load_dotenv()
FINNHUB_API = os.getenv("FINNHUB_API")
TICKER = "SPY"
START_DATE = "2025-04-06"
END_DATE = "2025-06-29"
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
print("Backtest Performance:", performance)

print("="*50)
plot_equity(prices, backtest_results["daily_pnls"], title=f"{TICKER} Strategy vs Benchmark")
plt.savefig(f"{TICKER} Strategy vs Benchmark")


try:
    result = quantstats_performance(backtest_results, prices)
    print(f"Function returned: {result}")
except Exception as e:
    print(f"ERROR in quantstats_performance: {e}")
    import traceback
    traceback.print_exc()

test_params = [
    (0.4, 0.2),   # Conservative
    (0.3, 0.15),  # Moderate
    (0.15, 0.05), # Aggressive
]

print("Threshold Testing Results:")
print("-" * 50)
for entry, exit in test_params:
    test_results = backtest_strategy(prices, combined_series, entry=entry, exit=exit, stoploss=0.1)
    total_return = (1 + test_results['daily_pnls']).prod() - 1
    sharpe = test_results['daily_pnls'].mean() / test_results['daily_pnls'].std() * np.sqrt(252)
    positions = test_results['positions']
    trading_days = (positions != 0).sum()
    print(f"Entry: {entry:4.2f}, Exit: {exit:4.2f} | Return: {total_return:6.1%} | Sharpe: {sharpe:5.2f} | Trading: {trading_days:2d} days")