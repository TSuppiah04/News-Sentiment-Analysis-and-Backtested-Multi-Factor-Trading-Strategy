import os
import pandas as pd
import requests
from dotenv import load_dotenv

"""
finnhub_news.py
Fetches financial news headlines from Finnhub API

Usage:
    from finnhub_news import fetch_news
    df = fetch_news("AAPL", "2023-01-01", "2023-01-31")
"""

load_dotenv()

def fetch_news(symbol, from_date=str, to_date=str) -> pd.DataFrame:
    FINNHUB_API = os.getenv("FINNHUB_API")

    if not FINNHUB_API:
        raise ValueError("FINNHUB_API key is not set. Please set it in your .env file.")
    url = (
    f"https://finnhub.io/api/v1/company-news"
    f"?symbol={symbol}&from={from_date}&to={to_date}&token={FINNHUB_API}"
    )

    try: 
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error fetching news: {e}")
    
    data = response.json()
    if not isinstance(data, list):
        raise ValueError(f"Unexpected response from API: {data}")
    
    df = pd.DataFrame(data)
    if df.empty:
        print("No news data found")
        return pd.DataFrame()
    
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], unit="s")
    else:
        df["datetime"] = pd.NaT

    if "headline" not in df.columns:
        df["headline"] = None
    if "summary" not in df.columns:
        df["summary"] = None

    return df[["datetime", "headline", "summary"]].sort_values("datetime").reset_index(drop=True)

if __name__ == "__main__":

    import datetime
    end = datetime.date.today()
    start = end - datetime.timedelta(days=7)

    print(f"Fetching news for SPY from {start} to {end}")
    news_df = fetch_news("SPY", start.isoformat(), end.isoformat())
    print(news_df.head())
