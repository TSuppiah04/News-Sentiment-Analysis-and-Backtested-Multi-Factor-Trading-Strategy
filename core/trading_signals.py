import numpy as np
import pandas as pd

def ensure_series(prices):
    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] == 1:
            return prices.iloc[:, 0]
        else:
            raise ValueError("Expected Series or single-column DataFrame")
    return prices


def momentum_signal(prices: pd.Series, window=10):
    prices = ensure_series(prices)
    ret = prices.pct_change(window)
    sig = np.sign(ret).fillna(0) 
    return sig

def volatility_signal(prices, window = 10):
    prices = ensure_series(prices)
    vol = prices.pct_change().rolling(window).std()
    vol_mean = vol.rolling(window).mean()
    sig = np.where(vol > vol_mean, 1, -1) 
    return pd.Series(sig, index=prices.index).fillna(0)

def reversion_signal(prices, window = 10, z_value = 1.0):
    prices = ensure_series(prices)
    rolling_mean = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std(ddof=0)
    z = (prices - rolling_mean) / rolling_std
    z = z.fillna(0)
    sig = z.apply(lambda x: 1 if x < -z_value else -1 if x > z_value else 0)
    return sig

def combined_signal(sentiment_series, momentum_series, volatility_series, reversion_series):

    idx = sentiment_series.index.union(momentum_series.index).union(volatility_series).union(reversion_series.index)
    s = sentiment_series.reindex(idx).fillna(method='ffill').fillna(0)
    m = momentum_series.reindex(idx).fillna(0)
    v = volatility_series.reindex(idx).fillna(0)
    r = reversion_series.reindex(idx).fillna(0)
    combined = 0.2*s + 0.3*m + 0.3*v + 0.2*r
    combined = combined.clip(-1, 1)
    return combined