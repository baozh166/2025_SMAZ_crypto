#!/usr/bin/env python
# coding: utf-8

"""
Show and save the latest new trades from the top-performing strategies
with costs, revenue details, and a log timestamp.
"""

import pandas as pd
import os
from datetime import datetime
from simulations import main

def show_latest_trades(n=10, top_k=3, output_file="latest_trades_log.csv"):
    """
    Show and save the latest n trades with positive predictions
    from the top_k performing strategies, appending to a running log.
    """
    # Run main() from simulations.py to get results and data
    results_df, data_df = main()

    # Pick top_k strategies by CAGR
    top_strategies = (
        results_df.sort_values("sim1_CAGR", ascending=False)
        .head(top_k)["prediction"]
        .tolist()
    )

    print(f"\nUsing top {top_k} strategies: {top_strategies}\n")

    # Collect trades where any top strategy fired (==1)
    trade_mask = data_df[top_strategies].sum(axis=1) > 0
    trades_df = data_df[trade_mask].copy()

    # Keep only the latest n trades (based on Date)
    latest_trades = trades_df.sort_values("Date", ascending=False).head(n)

    # Build display and log columns
    display_cols = ["Date", "Ticker", "Close"]
    log_cols = ["Date", "Ticker", "Close"]

    for strategy in top_strategies:
        prefix = strategy.split("_")[0]  # e.g. "predrf"
        display_cols += [strategy, f"sim1_gross_rev_{prefix}", f"sim1_fees_{prefix}", f"sim1_net_rev_{prefix}"]
        log_cols += [strategy, f"sim1_gross_rev_{prefix}", f"sim1_fees_{prefix}", f"sim1_net_rev_{prefix}"]

    latest_trades_out = latest_trades[log_cols].copy()

    # Add timestamp of when this batch was logged
    latest_trades_out["LoggedAt"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"Latest {n} new trades from top {top_k} strategies:\n")
    print(latest_trades_out)

    # Append to CSV (or create if not exists)
    if os.path.exists(output_file):
        latest_trades_out.to_csv(output_file, mode="a", header=False, index=False)
        print(f"\nAppended {len(latest_trades_out)} trades to: {output_file}")
    else:
        latest_trades_out.to_csv(output_file, index=False)
        print(f"\nCreated new log file: {output_file}")

    return latest_trades_out

if __name__ == "__main__":
    show_latest_trades(n=10, top_k=3, output_file="latest_trades_log.csv")
