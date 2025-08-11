from core.backtest import backtest_strategy, backtest_performance, plot_equity
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

def threshold_testing(prices, combined_series):
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