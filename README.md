# News-Sentiment-Analysis-and-Backtested-Multi-Factor-Trading-Strategy

# Overview:

This project executes a trading strategy combining 4 signals:

1. Momentum - Captures long-term strong/weak performance over a rolling window.
2. Mean Reversion - Trades against short-term deviations from rolling mean.
3. Volatility - Makes trades based off changes in rolling volatility. 
4. News Sentiment - Uses the NLP model finBERT to score news headlines. 

Using price data downloaded from Yahoo Finance, alongside fetching financial news from Finnhub, trading signals were generated and combined. Performance was then tested using backtesting. 

# Requirements:

1. Get free finnhub API key from https://finnhub.io/
2. Setup .env file and insert FINNHUB_API = (Your API key here)
3. Open terminal and input 'pip install -r requirements.txt

# Usage:

1. Adapt the 'TICKER', 'START_DATE', 'END_DATE' to what suits the user. 
2. Run main.py and change weights for combined signal alongside 'ENTRY_THRESHOLD', 'EXIT_THRESHOLD'.

# Results for SPY, AAPL AND TSLA:

Stock Comparison:

| Ticker        | CAGR           | Sharpe  | Avg Drawdown | 
| ------------- |:--------------:|:-------:|:------------:|
| SPY           | -16.18%        | -1.93   | -3.72%       | 
| AAPL          | 13.17%         |   0.77  | -9.36%       |
| MSFT          | -22.77%        |    -2.6 | -11.13%      |

Strategies vs Benchmark: 
