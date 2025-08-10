import quantstats as qs
import pandas as pd
import numpy as np
import os

def quantstats_performance(backtest_results, benchmark_results):
    strategy_returns = backtest_results['daily_pnls']
    benchmark_returns = benchmark_results.pct_change().fillna(0)

    dts = strategy_returns.index.intersection(benchmark_returns.index)
    strategy_returns = strategy_returns.loc[dts]
    benchmark_returns = benchmark_returns.loc[dts]

    qs.reports.html(strategy_returns, benchmark_returns, output="report.html")
    current_dir = os.getcwd()
    output_path = os.path.join(current_dir, "report.html")
    print(f"Saving report to: {output_path}")