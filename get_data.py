#!/usr/bin/env python
"""
Stock Data Collection Script
Fetches 3 crypto prices and define the target based on future growth > chg_threhold
"""

import numpy as np
import pandas as pd
import os, kagglehub
import yfinance as yf
from datetime import date

# Disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None

def get_1h_data_kg_yf(tickers):
    df_res = pd.DataFrame()

    for crypto in tickers: # ["BTC", "ETH", "XRP"]:
        # step 1. download data from kaggle
        path = kagglehub.dataset_download(f"imranbukhari/comprehensive-{crypto.lower()}usd-1h-data")
        
        # read the downloaded data as df
        df = pd.read_csv(os.path.join(path, f"{crypto}USD_1h_Combined_Index.csv"), index_col=0)
        
        # convert index to datetime and localize to UTC
        df.index = pd.to_datetime(df.index).tz_localize("UTC")

        # step 2. get the end time from the last index
        end_kagg_h = df.index[-1] # a UTC timestamp

        # YF only fetches from the beginning of the day
        end_kagg_day = str(end_kagg_h.date())
        historyPrices = yf.Ticker(f"{crypto}-USD").history(start=end_kagg_day, interval="1h")

        # Filter to start at 1 hour after the end_of_kaggle_hour
        start_yf_h = end_kagg_h + pd.Timedelta(hours=1)
        historyPrices = historyPrices[historyPrices.index >= start_yf_h]

        # Add symbol column for clarity
        df["Ticker"] = crypto
        historyPrices["Ticker"] = crypto

        # step 3. Concatenate and clean
        df = pd.concat([df, historyPrices]).sort_index().drop_duplicates()

        # step 4 merge to the fianl out df
        df_res = pd.concat([df_res, df])

    # drop the two columns not applicabel to Cryptos
    df_res.drop(columns=["Dividends", "Stock Splits"], inplace=True)
           
    return df_res


def set_target(df, forecast_period=24, chg_threshold = 0.01):

    df_tar = pd.DataFrame()
    
    tickers = df.Ticker.unique()
        
    """
    .rolling(window=forecast_period) 
    Creates a rolling window of PAST forecast_period. For each index 𝑡, it looks at the last forecast_period values (including the current one).

    .max()
    Within each rolling window, it computes the maximum closing price.
    So at time 𝑡, you get the maximum close over the past forecast_period bars.
    
    .shift(-(forecast_period - 1))
    Normally, rolling windows are backward‑looking (past values). By shifting forward (negative shift), you realign the maximum
    so that at time 𝑡, it corresponds to the future maximum over the next forecast_period bars.
    Specifically, shift(-(forecast_period - 1)) moves the max value forward so that the label at time 𝑡 
    is the maximum Close between 𝑡 and (𝑡 + 𝑓𝑜𝑟𝑒𝑐𝑎𝑠𝑡_𝑝𝑒𝑟𝑖𝑜𝑑 −1).
    """
    for ticker in tickers:
        df1 = df[df.Ticker == ticker]
        
        # get the future max price
        future_max_price = df1['Close'].rolling(window=forecast_period).max().shift(-(forecast_period - 1))
    
        # store as a new column
        df1[f'Max_Price_in_future_{forecast_period}h'] = future_max_price
    
        # Calculate the maximum potential growth (ratio)
        df1[f'growth_future_{forecast_period}h'] = df1[f'Max_Price_in_future_{forecast_period}h'] / df1['Close']
        
        # Assigns 1 if the value is greater than chg_threshold (e.g. > +1% change, change $40 out of $4000, ~50% data ), else assigns 0
        df1[f'is_positive_growth_{forecast_period}h_future'] = np.where(df1[f'growth_future_{forecast_period}h'] > 1 + chg_threshold, 1, 0)
        # The exit policy should match that objective: take profit as soon as +1% is available within the window, otherwise exit by timeout, 
        # with clear risk controls.

        # step 4 merge to the fianl out df
        df_tar= pd.concat([df_tar, df1])

    return df_tar
        

def save_data(df):
    """Save data to parquet files"""
    data_dir = os.path.join(os.getcwd(), "data")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    today = date.today().strftime("%Y%m%d")
    
    if not df.empty:
        file_path = os.path.join(data_dir, f"{today}_df_wTarget.parquet")
        try:
            df.to_parquet(file_path, compression="brotli")
            print(f"df_tar Saved to {file_path}")
        except Exception as e:
            print(f"Error saving df_tar: {e}")
    else:
        print(f"Warning: df_tar is empty, not saving")


def main():
    """Main execution function"""
    print("Starting data collection process...")

    tickers = ["BTC", "ETH", "XRP"]
    
    # Fetch data
    print("\n=== Fetching Stock Data ===")
    stocks_df = get_1h_data_kg_yf(tickers)

    # create target
    stocks_df = set_target(stocks_df, forecast_period=24, chg_threshold = 0.01)
    
    # Save data
    print("\n=== Saving Data ===")
    save_data(stocks_df)
    
    print("Data collection completed!")
    return stocks_df


if __name__ == "__main__":
    stocks_df = main()
