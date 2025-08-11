import numpy as np
import pandas as pd
from core.backtest import backtest_strategy, backtest_performance, plot_equity

def walk_forward(prices, signals, window = 60, step = 20, entry=[0.2,0.3,0.4], exit=[0.1, 0.2], stoploss=0.1):
    results = []
    for start in range(0, len(prices) - window, step):
        train = slice(start, start + window)
        test = slice(start + window, min(start + window + step, len(prices)))

        best_sharpe = -np.inf
        best_entry, best_exit = entry, exit

        for e in entry: 
            for x in exit: 
                res_train = backtest_strategy(prices.iloc[train], signals.iloc[train], entry=e, exit=x, stoploss=stoploss)
                perf_train = backtest_performance(res_train['daily_pnls'])
                if perf_train['sharpe_ratio'] > best_sharpe:
                    best_sharpe = perf_train['sharpe_ratio']
                    best_entry, best_exit = e, x

        res_test = backtest_strategy(prices.iloc[test], signals.iloc[test], entry=best_entry, exit=best_exit, stoploss=stoploss)
        results.append(res_test['daily_pnls'])

    return pd.concat(results)

