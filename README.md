# design 
Notebooks for experiments and cleaning up before uploading
scripts for automation

# names of Notebooks + scripts
* get_data
* tranform_data(feature Engineering)
* Modeling
* Simulation
* Advanced_simulation
* Automation

# get_data
BTC, ETH, XRP 1h data from kaggle + yf, ensuring continuity. 
Target definition: The future_max_price + ratio + binary label (future_growth_ratio > chg_threhold as 1) is a clean way to generate supervised learning targets.
Saving to parquet with compression makes downstream ML pipelines efficient.


# tranform_data(feature Engineering)
custom features, like SMA EMA crossing touching
All TAlib technique indicators(should understand every single TA)

# modeling
4 models(decision Tree, RF, XGBoost, LSTM - time consuming due to large data)


# simulation
buy low, fees, stop loss, take profit, or exit at the 24th h.
CAGR

# TODO Automation
sending Telegram/email notification




