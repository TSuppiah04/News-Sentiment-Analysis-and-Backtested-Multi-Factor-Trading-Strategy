def momentum_signal(prices, window = 10):
    ret = prices.pct_change(window)
    sig = ret.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    return sig.fillna(0)

def volatility_signal(prices, window = 10):
    vol = prices.pct_change().rolling(window).std()
    sig = vol.apply(lambda x: 1 if x > vol.mean() else -1 if x < vol.mean() else 0)
    return sig.fillna(0)

def reversion_signal(prices, window = 10, z_value = 1.0):
    rolling_mean = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std(ddof=0)
    z = (prices - rolling_mean) / rolling_std
    z = z.fillna(0)
    sig = z.apply(lambda x: 1 if x < -z_value else -1 if x > z_value else 0)
    return sig

def volume_signal(sentiment_series, momentum_series, volatility_series, reversion_series):
    idx = sentiment_series.index.union(momentum_series.index).union(volatility_series).union(reversion_series.index)
    s = sentiment_series.reindex(idx).fillna(0)
    m = momentum_series.reindex(idx).fillna(0)
    v = volatility_series.reindex(idx).fillna(0)
    r = reversion_series.reindex(idx).fillna(0)
    combined = 0.2*s + 0.3*m + 0.3*v + 0.2*r
    max_val = 1
    combined = combined/max_val
    combined = combined.clip(-1, 1)
    return combined