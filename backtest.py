import pandas as pd
import numpy as np

def backtest_strategy(prices: pd.Series, signals: pd.Series, entry: float = 0.25, exit: float = 0.0, stoploss: float = None) -> pd.DataFrame:

    prices = prices.copy()
    signals = signals.copy()

    inpos = 0
    entry_price = None
    daily_pnls = []
    positions = []
    entry_prices = []

    for i in enumerate(prices):
        price = prices.iloc[i]
        signal = signals.iloc[i]

        if inpos == 0: 
            if signal > entry:
                inpos = 1
                entry_price = price

            elif signal < -entry:
                inpos = -1
                entry_price = price

        elif inpos == 1:
            if signal < exit:
                inpos = 0
                entry_price = None
            elif stoploss is not None and (price - entry_price) / entry_price < -stoploss:
                inpos = 0
                entry_price = None

        
        if i > 0:
            daily_return = (price - prices.iloc[i-1]) / prices.iloc[i-1]
            daily_pnl = inpos * daily_return
        else:
            daily_pnl = 0

        positions.append(inpos)
        entry_prices.append(entry_price if entry_price is not None else np.nan)
        daily_pnls.append(daily_pnl)

    results = pd.DataFrame({
        "positions": positions,
        "entry_prices": entry_prices,
        "daily_pnls": daily_pnls
    }, index=prices.index)

    return results       

def backtest_performance(daily_pnls: pd.Series) -> dict:
    daily_pnls = pd.Series(daily_pnls).fillna(0)
    cumulative_return = (1 + daily_pnls).cumprod() - 1
    annual_return = (1 + cumulative_return) ** (252 / len(daily_pnls)) - 1
    annual_volatility = daily_pnls.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else np.nan