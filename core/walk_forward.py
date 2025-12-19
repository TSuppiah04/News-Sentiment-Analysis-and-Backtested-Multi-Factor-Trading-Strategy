import numpy as np
import pandas as pd
from core.backtest import backtest_strategy, backtest_performance, plot_equity

def walk_forward(prices, signals, window = 60, step = 20, entry=[0.2,0.3,0.4,0.5], exit=[0.1, 0.2,0.3], stoploss=0.1):
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
        daily_pnls = res_test['daily_pnls']
        if isinstance(daily_pnls, pd.DataFrame):
            daily_pnls = daily_pnls.sum(axis=1)
        

        results.append(pd.Series(daily_pnls.values.astype(float), index=daily_pnls.index))


    all_pnls = pd.concat(results)
    all_pnls = pd.to_numeric(all_pnls, errors='coerce')
    all_pnls = all_pnls.groupby(all_pnls.index).sum()

    return {"daily_pnls": all_pnls}

