import os
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
FINNHUB_API = os.getenv("FINNHUB_API")

def fetch_news(symbol, category=str, from_date=str, to_date=str) -> pd.DataFrame:
    url = (
    f"https://finnhub.io/api/v1/company-news"
    f"?symbol={ticker}&from={start_date}&to={end_date}&token={FINNHUB_API_KEY}"
    )

    try: 
        response = responses.get(url)
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