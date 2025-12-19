import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quantstats as qs

def backtest_strategy(prices: pd.Series, signals: pd.Series, entry: float = 0.25, exit: float = 0.15, stoploss: float = None) -> pd.DataFrame:

    prices = prices.copy()
    signals = signals.shift(1).fillna(0)

    inpos = 0
    entry_price = None
    daily_pnls = []
    positions = []
    entry_prices = []

    for i in range(len(prices)):
        price = prices.iloc[i]
        signal = signals.iloc[i]
        prev_pos = inpos

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
            elif stoploss is not None and entry_price is not None and (price - entry_price) / entry_price < -stoploss:
                inpos = 0
                entry_price = None

        elif inpos == -1:
            if signal > -exit:
                inpos = 0
                entry_price = None
            elif stoploss is not None and entry_price is not None and (entry_price - price) / entry_price < -stoploss:
                inpos = 0
                entry_price = None

        if i > 0:
            daily_return = (price - prices.iloc[i-1]) / prices.iloc[i-1]
            daily_pnl = prev_pos * daily_return
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
    cumulative_return = (1 + daily_pnls).prod()
    total_return = cumulative_return - 1
    annual_return = (1 + total_return) ** (252 / len(daily_pnls)) - 1
    annual_volatility = daily_pnls.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else np.nan

    return {
        "cumulative_return": cumulative_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio

    }

def plot_equity(prices: pd.Series, strategy_results: pd.Series, title: str = "Strategy vs Benchmark"):
    strategy_cum = (1 + strategy_results).cumprod()
    benchmark_results = prices.pct_change().fillna(0)
    benchmark_cum = (1 + benchmark_results).cumprod()
    
    plt.figure(figsize=(14, 7))
    plt.plot(strategy_cum.index, strategy_cum.values, label = "Strategy", color='blue', alpha=0.7)
    plt.plot(benchmark_cum.index, benchmark_cum.values, label = "Benchmark", color='orange', alpha=0.7)
    plt.legend()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"images/equity_curve_{title}.png", dpi=300, bbox_inches='tight')
    plt.show()