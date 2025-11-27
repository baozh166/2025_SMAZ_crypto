"""
Simulations with different trading parameters - Standalone Version
referenced from simulations.py
"""

from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import os
import joblib, random
from datetime import date

window=24
# Define a function to get the rolling max High and min Low for the next 30 trading days
# Sort the DataFrame: Sorting by Ticker and Date ensures that we are looking at each ticker's data in chronological order.
# Rolling window calculation: We use the rolling method with a window of 5 to calculate the maximum high and minimum low over the next 5 days.
# The shift method is used to align these values correctly with the current row.
def rolling_max_min(df, window=window):
    # high/low in 30 days
    df[f'Max_High_Next_{window}'] = df['High'].rolling(window=window, min_periods=1).max().shift(-window+1)
    df[f'Min_Low_Next_{window}'] = df['Low'].rolling(window=window, min_periods=1).min().shift(-window+1)

    # low in 1 day (for lower entry)
    df['Min_Low_Next_1'] = df['Low'].rolling(window=1, min_periods=1).min().shift(-1)
    return df
    

def load_data(window=window):
    """Load the processed data from joblib file, and process it for advanced simulations"""
    data_dir = os.getcwd() + "/data"
    model_file_name = "new_df_wProbability_entrySignals.joblib"
    path = os.path.join(data_dir, model_file_name)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")

    new_df = joblib.load(path)
   
    # rank of prediction for Advanced simulation
    new_df["pred10_rank"] = new_df.groupby("Date")["proba_pred10"].rank(method="first", ascending=False)

    # Apply the function to each group of Ticker
    # Important!: need to drop index from groupby operation (reset_index) - so that you can merge that afterwards
    result = new_df[new_df.split=='test'][['Date','High','Low','Close','Ticker']].groupby('Ticker').apply(rolling_max_min).reset_index(drop=True)

    # Calculate the ratios + safe divide
    result[f'Ratio_MaxHighNext{window}_to_Close'] = np.where(result['Close'] == 0, # condition
                                                             np.nan,  # value if condition is True
                                                             result[f'Max_High_Next_{window}']/result['Close'] # value if condition is False
                                                            )
    # more idiomatic Pandas way using .where()
    result[f'Ratio_MinLowNext{window}_to_Close'] = (result[f'Min_Low_Next_{window}']/result['Close']).where(result['Close'] != 0)
    
    result['Ratio_MinLowNext1_to_Close'] = np.where(result['Close'] == 0, np.nan,  result['Min_Low_Next_1']/result['Close'])

    # Merge the results back with the original dataframe
    new_df = new_df.merge(result[['Date', 'Ticker', f'Max_High_Next_{window}', f'Min_Low_Next_{window}',
                                  f'Ratio_MaxHighNext{window}_to_Close', f'Ratio_MinLowNext{window}_to_Close','Ratio_MinLowNext1_to_Close']], 
                          on=['Date', 'Ticker'])

    print(f"Data loaded successfully. Shape: {new_df.shape}")
    
    return new_df
    

# define a class for Simulation Parameters
@dataclass
class SimulationParams:
    initial_capital: float
    fees: float
    portfolio_optimization: bool
    threshold: float
    top_k: int
    stop_loss: float
    take_profit: float
    lower_entry: float

# future_gross_return, depending on lower_entry, take_profit, stop_loss
def get_future_gross_return(row, sim_params:SimulationParams):
    if row['lower_entry']==0: # no trade, investment is untouched, no fees
        return row['investment']

  # buy trade is filled for ALL next cases:
    if row['take_profit']==1 and row['stop_loss']==1:
        if random.random()>0.5: #assume take_profit event was first
            return  row['investment']*(sim_params.take_profit+(1-sim_params.lower_entry))
        else: #assume stop_loss event was first
            return row['investment']*(sim_params.stop_loss+(1-sim_params.lower_entry))

    if row['take_profit']==1: # take some good profit, pay fees
        return  row['investment']*(sim_params.take_profit+(1-sim_params.lower_entry))

    if row['stop_loss']==1: # fix the loss, pay fees
        return row['investment']*(sim_params.stop_loss+(1-sim_params.lower_entry))

    # no stop_loss and no take_profit
    if pd.isna(row['growth_future_24h']):
        return row['investment'] # no information on growth in 30 days --> return the same investment in 5 days
    else:
        # EXIT at the maximum Close within 24h! Tricky to execute in practical!
        return row['investment']*(row['growth_future_24h']+(1-sim_params.lower_entry))

    
# fees, depending on lower_entry, take_profit, stop_loss
def get_fees(row, sim_params:SimulationParams):
    if row['lower_entry']==0: # no trade ==> no fees
        return 0

    # pay fees in all other cases
    return -row['investment']*sim_params.fees

    
# define a one day simulation
def one_date_simulation(date:str, invest_sum:float, df:pd.DataFrame, sim_params:SimulationParams, predictor:str='proba_pred10', window:int=window):

    rank_column = predictor.split('_')[1]+'_rank' # e.g. 'proba_pred10' --> 'pred10_rank'
    
    # 1. get TOP_K (or ALL) predictions from the predictor (pred14_rf_best_rule_53 by default), that are higher than THE THRESHOLD
    if sim_params.top_k is None:
        one_day_predictions_df = df[(df.Date==date)&(df[predictor] > sim_params.threshold)]
    else:
        one_day_predictions_df = df[(df.Date==date)&(df[predictor] > sim_params.threshold)&(df[rank_column]<=sim_params.top_k)]
    
    FIELDS = ['Close', 'Ticker', 'Date', predictor, rank_column, 'growth_future_24h', # useing Close instead of Close_x ZB 11032025
            f'Ratio_MaxHighNext{window}_to_Close', f'Ratio_MinLowNext{window}_to_Close','Ratio_MinLowNext1_to_Close']
    result_df = one_day_predictions_df[FIELDS].copy()
    
    # 2. Get non-normalized weights: probability-threshold + 0.01
    result_df['weight'] = result_df[predictor] - sim_params.threshold +0.01
    
    # 3. Get normalized weights
    result_df['weight_norm'] = result_df['weight']/result_df['weight'].sum()
    
    # 4. Make bets to allocate 'invest_sum' across all suitable predictions
    result_df['investment'] = result_df['weight_norm'] * invest_sum
    
    # 5. Lower Entry: the trade is executed only is Low price for next day is lower than the bet (Adj_Close_today * sim_params.lower_entry)
    # [ONLY TRADES with lower_entry==1 are filled by the exchange]
    result_df['lower_entry'] = (result_df['Ratio_MinLowNext1_to_Close'] <= sim_params.lower_entry).astype(int)
    
    # 6. Stop Loss: happens if the current price (or Low price) goes below stop loss threshold during one of the next 5 periods (1 week)
    result_df['stop_loss'] = (result_df[f'Ratio_MinLowNext{window}_to_Close'] <= sim_params.stop_loss).astype(int)
    
    # 7. Take Profit: take the money if the current Price (or Max_price) goes higher than sim_params.take_profit
    result_df['take_profit'] = (result_df[f'Ratio_MaxHighNext{window}_to_Close'] >= sim_params.take_profit).astype(int)
    
    # 8. Calculate future returns (when the order is executed + stop_loss True/False + take_profit True/False)
    result_df['future_gross_return'] = result_df.apply(lambda row: get_future_gross_return(row,sim_params=sim_params), axis=1)
    result_df['fees'] =  result_df.apply(lambda row: get_fees(row,sim_params=sim_params), axis=1)
    result_df['future_net_return'] = result_df['future_gross_return'] + result_df['fees']
    
    return result_df


# Generate fin result for ALL days
def simulate(df:pd.DataFrame, sim_params:SimulationParams, window:int=window):

    simulation_df = None
    
    # all dates for simulation
    all_dates = df[df.split=='test'].sort_values(by='Date').Date.unique()
    
    # arrays of dates and capital available (capital for the first 5 days)
    dates = []
    capital= window * [sim_params.initial_capital/window]  # first 30 periods trade with 1/30 of the initial_capital. e.g. [333,...,333] = 10k in total
    
    for current_date in all_dates[0:-window]:  #growth_future_30d is not defined for the last 30 days : ALL, but last 30 dates
    
        current_invest_sum = capital[-window]    # take the value or everything that you can sell from 30 days ago
        
        one_day_simulation_results = one_date_simulation(date = current_date,  # one day simulation result
                                        invest_sum = current_invest_sum,
                                        df = df,
                                        sim_params=sim_params,
                                        predictor='proba_pred10')
        
        # add capital available in 30 days
        if len(one_day_simulation_results)==0:  #no predictions -> no trades
            capital.append(current_invest_sum)
        else:
            capital.append(one_day_simulation_results.future_net_return.sum())
        dates.append(current_date)
        
        if simulation_df is None:
            simulation_df = one_day_simulation_results
        else:
            simulation_df = pd.concat([simulation_df, one_day_simulation_results], ignore_index=True)
    
    # add last 5 days to make the count of data points equal for dates/capital arrays
    dates.extend(all_dates[-window:])
    capital_df = pd.DataFrame({'capital':capital}, index=pd.to_datetime(dates))


    # Convert dataclass instance to dictionary
    sim_params_dict = asdict(sim_params)
    # Create a single-row DataFrame
    sim_params_df = pd.DataFrame([sim_params_dict])

    
    # calculate the financial results for this set of sim_params
    fin_res = {}
    
    # Basic counts
    fin_res["total_bids"] = len(simulation_df)
    fin_res["avg_bids_daily"] = len(simulation_df) / simulation_df.Date.nunique()
    fin_res["filled_bids"] = len(simulation_df[simulation_df.lower_entry == 1])
    fin_res["fill_bids_percent"] = fin_res["filled_bids"] / fin_res["total_bids"]
    
    # Stop loss stats
    stop_loss_filter = (simulation_df.lower_entry == 1) & (simulation_df.stop_loss == 1)
    fin_res["stop_loss_count"] = len(simulation_df[stop_loss_filter])
    fin_res["stop_loss_net"] = (
        simulation_df[stop_loss_filter].future_net_return.sum()
        - simulation_df[stop_loss_filter].investment.sum()
    )
    
    # Take profit stats
    take_profit_filter = (simulation_df.lower_entry == 1) & (simulation_df.take_profit == 1)
    fin_res["take_profit_count"] = len(simulation_df[take_profit_filter])
    fin_res["take_profit_net"] = (
        simulation_df[take_profit_filter].future_net_return.sum()
        - simulation_df[take_profit_filter].investment.sum()
    )
    
    # Capital and CAGR
    fin_res["start_capital"] = sim_params.initial_capital
    fin_res["resulting_capital"] = capital_df[-window:].capital.sum()
    fin_res["cagr_factor"] = np.round((fin_res["resulting_capital"] / sim_params.initial_capital) ** (1 / 1.325), 3)
    fin_res["cagr_percent"] = np.round((fin_res["cagr_factor"] - 1) * 100.0, 2)
    
    # Convert to DataFrame
    fin_df = pd.DataFrame([fin_res])


    # # Concatenate side by side (columns)
    fin_df = pd.concat([sim_params_df.reset_index(drop=True), fin_df.reset_index(drop=True)], axis=1)
    
    
    return simulation_df, capital_df, fin_df




def main():
    all_params_combinations_df = pd.DataFrame()

    # Load data
    new_df = load_data()
    for i in new_df.columns:
        print(i)

    # consider Greedy Search and Bayesian optimization to shorten the computing times with trade‑offs in finding global optima
    for lower_entry in [0.98, 0.99, 1, 1.01]: # [0.97, 0.98, 0.99, 1, 1.01, 1.02, 1.03]:
        print("lower_entry:", lower_entry)
        for take_profit in [1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5]:
            print("take_profit:", take_profit)
            for stop_loss in [0.7, 0.75, 0.8, 0.85, 0.9]: # [0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.82, 0.84, 0.86, 0.9]: 
                print("stop_loss:", stop_loss)
                for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
                    print("threshold:", threshold)
                    for top_k in [1, 2, 3]: # [1, 2, 3, 4, 5, 6, 8, 10, 20, 33]:
                        print("top_k:", top_k)

                        # One simulation
                        sim_params = SimulationParams(
                            initial_capital=10000,
                            threshold=threshold,
                            fees=0.002,
                            top_k=top_k,
                            portfolio_optimization=False,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            lower_entry=lower_entry
                        )

                        # simulate() should return (simulation_df, capital_df, fin_metrics_df)
                        _, _, fin_metrics_df = simulate(new_df, sim_params)
                        print(fin_metrics_df)

                        all_params_combinations_df = pd.concat(
                            [all_params_combinations_df, fin_metrics_df],
                            ignore_index=True
                        )

    # sort by cagr_percent in place
    all_params_combinations_df.sort_values(by="cagr_percent", ascending=False, inplace=True)

    # prepare saving
    today = date.today().strftime("%Y%m%d")
    out_dir = "result_out"
    os.makedirs(out_dir, exist_ok=True)

    # Save results
    output_file = f'{out_dir}/{today}_Advanced_simu_results.csv'
    all_params_combinations_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    return all_params_combinations_df


if __name__ == "__main__":
    main()

                     
                        



    

