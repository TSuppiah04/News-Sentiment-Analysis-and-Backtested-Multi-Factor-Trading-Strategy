import quantstats as qs
import pandas as pd
import numpy as np
from datetime import datetime
import os

def quantstats_performance(strategy_results, benchmark_results, report_name=None):
    if isinstance(strategy_results, dict) and "daily_pnls" in strategy_results:
        strategy_returns = strategy_results['daily_pnls']
    elif isinstance(strategy_results, pd.Series):
        strategy_returns = strategy_results
    elif isinstance(strategy_results, pd.DataFrame):
        strategy_returns = strategy_results["daily_pnls"]
    else:
        raise ValueError("Invalid strategy results format")
    
    strategy_returns.index = pd.to_datetime(strategy_returns.index)

    benchmark_returns = benchmark_results.pct_change().fillna(0)

    dts = strategy_returns.index.intersection(benchmark_returns.index)
    strategy_returns = strategy_returns.loc[dts]
    benchmark_returns = benchmark_returns.loc[dts]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if report_name is None:
        filename = f"report_{timestamp}.html"
    else:
        filename = f"{report_name}_{timestamp}.html"

    qs.reports.html(strategy_returns, benchmark_returns, output=filename)
    current_dir = os.getcwd()
    output_path = os.path.join(current_dir, filename)
    print(f"Saving report to: {output_path}")