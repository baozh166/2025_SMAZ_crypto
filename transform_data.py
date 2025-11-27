#!/usr/bin/env python
"""
Data Transformation Script
Adds technical indicators using TA-Lib and merges datasets
"""

import pandas as pd
import talib
import os
import numpy as np
from datetime import date

# Disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None


def load_data():
    """Load data from parquet files"""
    data_dir = os.path.join(os.getcwd(), "data")

    try:
        stocks_df = pd.read_parquet(os.path.join(data_dir, "20251113_df_wTarget.parquet"))

        print(f"Loaded data - Stocks: {stocks_df.shape}")
        return stocks_df

    except FileNotFoundError as e:
        print(f"Error: Required data files not found: {e}")
        print("Please run get_data.py first to generate the required data files")
        return None
        

def prepare_data(stocks_df):
    """Prepare data for transformation"""
    # Convert Volume to float
    stocks_df['Volume'] = stocks_df['Volume'] * 1.0

    # Convert required columns to float64 for TA-Lib
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        stocks_df.loc[:, col] = stocks_df.loc[:, col].astype('float64')

    # Truncate data to start from 2017
    cutoff_date = pd.to_datetime("2017-01-01").tz_localize("UTC")
    stocks_df = stocks_df[stocks_df.index >= cutoff_date]

    return stocks_df


def custom_features_for_one_ticker(df, forecast_period=24, scale_hours=720):
    # add DATE features
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Weekday'] = df.index.weekday
    df['Date'] = df.index.date  # same type as datetime.date(2024, 1, 4)
    
    # Historical returns
    for period in [1, 4, 8, 24, 120, 360]:
        df[f'his_growth_{period}h'] = df['Close'] / df['Close'].shift(period)
    
    # Technical indicators
    # Exponential Moving Averages for 10, 20, SimpleMovingAverage 50, 200 h
    df['EMA10']  = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA20']  = df['Close'].ewm(span=20, adjust=False).mean()
    df['SMA50']  = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()

    # derivations
    df['growing_moving_average'] = np.where(df['EMA10'] > df['EMA20'], 1, 0)
    df['high_minus_low_relative'] = (df['High'] - df['Low']) / df['Close']
    df['high_touching_SMA50_relative'] = (df['High'] - df['SMA50']) / df['Close']
    df['low_touching_SMA50_relative']  = (df['Low']  - df['SMA50']) / df['Close']
    df['high_touching_SMA200_relative'] = (df['High'] - df['SMA200']) / df['Close']
    df['low_touching_SMA200_relative']  = (df['Low']  - df['SMA200']) / df['Close']
    
    # volatility is usually defined on returns, not raw prices.
    df['returns'] = df['Close'].pct_change()
    
    # n periods rolling monthlized hourly volatility 24*30=720
    df['volatility'] = df['returns'].rolling(forecast_period).std() * np.sqrt(scale_hours) # annualized 24*365=8760
  
    return df


def get_custom_features(stocks_df):

    ALL_TICKERS = stocks_df["Ticker"].unique().tolist()
    print(ALL_TICKERS)
    
    # adding Momentum / Pattern/ Volume features to all tickers - one by one
    results = []
    
    i=0
    for ticker in ALL_TICKERS: # MUST do it one by one!! ZB20251105
        i+=1
        print(f'{i}/{len(ALL_TICKERS)} Current ticker is {ticker}')
        current_ticker_df = stocks_df[stocks_df.Ticker == ticker]
    
        # get additional features
        df_custom_features = custom_features_for_one_ticker(current_ticker_df)

        # add to the result list for concating
        results.append(df_custom_features)

    # merge to the output df
    res_df = pd.concat(results, # a list of 3 dfs
                       axis = 0 # extend rows
                      )
    
    return res_df

    
def talib_get_momentum_indicators_for_one_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate TA-Lib momentum indicators for one ticker"""
    try:
        # ADX - Average Directional Movement Index
        talib_momentum_adx = talib.ADX(df.High.values, df.Low.values, df.Close.values, timeperiod=14)
        # ADXR - Average Directional Movement Index Rating
        talib_momentum_adxr = talib.ADXR(df.High.values, df.Low.values, df.Close.values, timeperiod=14)
        # APO - Absolute Price Oscillator
        talib_momentum_apo = talib.APO(df.Close.values, fastperiod=12, slowperiod=26, matype=0)
        # AROON - Aroon
        talib_momentum_aroon = talib.AROON(df.High.values, df.Low.values, timeperiod=14)
        # AROONOSC - Aroon Oscillator
        talib_momentum_aroonosc = talib.AROONOSC(df.High.values, df.Low.values, timeperiod=14)
        # BOP - Balance of Power
        talib_momentum_bop = talib.BOP(df.Open.values, df.High.values, df.Low.values, df.Close.values)
        # CCI - Commodity Channel Index
        talib_momentum_cci = talib.CCI(df.High.values, df.Low.values, df.Close.values, timeperiod=14)
        # CMO - Chande Momentum Oscillator
        talib_momentum_cmo = talib.CMO(df.Close.values, timeperiod=14)
        # DX - Directional Movement Index
        talib_momentum_dx = talib.DX(df.High.values, df.Low.values, df.Close.values, timeperiod=14)
        # MACD - Moving Average Convergence/Divergence
        talib_momentum_macd, talib_momentum_macdsignal, talib_momentum_macdhist = talib.MACD(
            df.Close.values, fastperiod=12, slowperiod=26, signalperiod=9)
        # MACDEXT - MACD with controllable MA type
        talib_momentum_macd_ext, talib_momentum_macdsignal_ext, talib_momentum_macdhist_ext = talib.MACDEXT(
            df.Close.values, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
        # MACDFIX - Moving Average Convergence/Divergence Fix 12/26
        talib_momentum_macd_fix, talib_momentum_macdsignal_fix, talib_momentum_macdhist_fix = talib.MACDFIX(
            df.Close.values, signalperiod=9)
        # MFI - Money Flow Index
        talib_momentum_mfi = talib.MFI(df.High.values, df.Low.values, df.Close.values, df.Volume.values, timeperiod=14)
        # MINUS_DI - Minus Directional Indicator
        talib_momentum_minus_di = talib.MINUS_DM(df.High.values, df.Low.values, timeperiod=14)
        # MOM - Momentum
        talib_momentum_mom = talib.MOM(df.Close.values, timeperiod=10)
        # PLUS_DI - Plus Directional Indicator
        talib_momentum_plus_di = talib.PLUS_DI(df.High.values, df.Low.values, df.Close.values, timeperiod=14)
        # PLUS_DM - Plus Directional Movement
        talib_momentum_plus_dm = talib.PLUS_DM(df.High.values, df.Low.values, timeperiod=14)
        # PPO - Percentage Price Oscillator
        talib_momentum_ppo = talib.PPO(df.Close.values, fastperiod=12, slowperiod=26, matype=0)
        # ROC - Rate of change
        talib_momentum_roc = talib.ROC(df.Close.values, timeperiod=10)
        # ROCP - Rate of change Percentage
        talib_momentum_rocp = talib.ROCP(df.Close.values, timeperiod=10)
        # ROCR - Rate of change ratio
        talib_momentum_rocr = talib.ROCR(df.Close.values, timeperiod=10)
        # ROCR100 - Rate of change ratio 100 scale
        talib_momentum_rocr100 = talib.ROCR100(df.Close.values, timeperiod=10)
        # RSI - Relative Strength Index
        talib_momentum_rsi = talib.RSI(df.Close.values, timeperiod=14)
        # STOCH - Stochastic
        talib_momentum_slowk, talib_momentum_slowd = talib.STOCH(
            df.High.values, df.Low.values, df.Close.values,
            fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        # STOCHF - Stochastic Fast
        talib_momentum_fastk, talib_momentum_fastd = talib.STOCHF(
            df.High.values, df.Low.values, df.Close.values,
            fastk_period=5, fastd_period=3, fastd_matype=0)
        # STOCHRSI - Stochastic Relative Strength Index
        talib_momentum_fastk_rsi, talib_momentum_fastd_rsi = talib.STOCHRSI(
            df.Close.values, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        # TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
        talib_momentum_trix = talib.TRIX(df.Close.values, timeperiod=30)
        # ULTOSC - Ultimate Oscillator
        talib_momentum_ultosc = talib.ULTOSC(
            df.High.values, df.Low.values, df.Close.values, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        # WILLR - Williams' %R
        talib_momentum_willr = talib.WILLR(df.High.values, df.Low.values, df.Close.values, timeperiod=14)

        momentum_df = pd.DataFrame({
            # 'Date': df.Date.values,
            # 'Ticker': df.Ticker,
            'adx': talib_momentum_adx,
            'adxr': talib_momentum_adxr,
            'apo': talib_momentum_apo,
            'aroon_1': talib_momentum_aroon[0],
            'aroon_2': talib_momentum_aroon[1],
            'aroonosc': talib_momentum_aroonosc,
            'bop': talib_momentum_bop,
            'cci': talib_momentum_cci,
            'cmo': talib_momentum_cmo,
            'dx': talib_momentum_dx,
            'macd': talib_momentum_macd,
            'macdsignal': talib_momentum_macdsignal,
            'macdhist': talib_momentum_macdhist,
            'macd_ext': talib_momentum_macd_ext,
            'macdsignal_ext': talib_momentum_macdsignal_ext,
            'macdhist_ext': talib_momentum_macdhist_ext,
            'macd_fix': talib_momentum_macd_fix,
            'macdsignal_fix': talib_momentum_macdsignal_fix,
            'macdhist_fix': talib_momentum_macdhist_fix,
            'mfi': talib_momentum_mfi,
            'minus_di': talib_momentum_minus_di,
            'mom': talib_momentum_mom,
            'plus_di': talib_momentum_plus_di,
            'dm': talib_momentum_plus_dm,
            'ppo': talib_momentum_ppo,
            'roc': talib_momentum_roc,
            'rocp': talib_momentum_rocp,
            'rocr': talib_momentum_rocr,
            'rocr100': talib_momentum_rocr100,
            'rsi': talib_momentum_rsi,
            'slowk': talib_momentum_slowk,
            'slowd': talib_momentum_slowd,
            'fastk': talib_momentum_fastk,
            'fastd': talib_momentum_fastd,
            'fastk_rsi': talib_momentum_fastk_rsi,
            'fastd_rsi': talib_momentum_fastd_rsi,
            'trix': talib_momentum_trix,
            'ultosc': talib_momentum_ultosc,
            'willr': talib_momentum_willr,
            },
            index = df.index # keep the same index for pd.concat() in the next step
        )

        return momentum_df

    except Exception as e:
        print(f"Error calculating momentum indicators: {e}")
        return pd.DataFrame()


def talib_get_volume_volatility_cycle_price_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # TA-Lib Volume indicators
    # https://github.com/TA-Lib/ta-lib-python/blob/master/docs/func_groups/volume_indicators.md
    # AD - Chaikin A/D Line
    talib_ad = talib.AD(
        df.High.values, df.Low.values, df.Close.values, df.Volume.values)
    # ADOSC - Chaikin A/D Oscillator
    talib_adosc = talib.ADOSC(
        df.High.values, df.Low.values, df.Close.values, df.Volume.values, fastperiod=3, slowperiod=10)
    # OBV - On Balance Volume
    talib_obv = talib.OBV(
        df.Close.values, df.Volume.values)

    # TA-Lib Volatility indicators
    # https://github.com/TA-Lib/ta-lib-python/blob/master/docs/func_groups/volatility_indicators.md
    # ATR - Average True Range
    talib_atr = talib.ATR(
        df.High.values, df.Low.values, df.Close.values, timeperiod=14)
    # NATR - Normalized Average True Range
    talib_natr = talib.NATR(
        df.High.values, df.Low.values, df.Close.values, timeperiod=14)

    # TA-Lib Cycle Indicators
    # https://github.com/TA-Lib/ta-lib-python/blob/master/docs/func_groups/cycle_indicators.md
    # HT_DCPERIOD - Hilbert Transform - Dominant Cycle Period
    talib_ht_dcperiod = talib.HT_DCPERIOD(df.Close.values)
    # HT_DCPHASE - Hilbert Transform - Dominant Cycle Phase
    talib_ht_dcphase = talib.HT_DCPHASE(df.Close.values)
    # HT_PHASOR - Hilbert Transform - Phasor Components
    talib_ht_phasor_inphase, talib_ht_phasor_quadrature = talib.HT_PHASOR(
        df.Close.values)
    # HT_SINE - Hilbert Transform - SineWave
    talib_ht_sine_sine, talib_ht_sine_leadsine = talib.HT_SINE(
        df.Close.values)
    # HT_TRENDMODE - Hilbert Transform - Trend vs Cycle Mode
    talib_ht_trendmode = talib.HT_TRENDMODE(df.Close.values)

    # TA-Lib Price Transform Functions
    # https://github.com/TA-Lib/ta-lib-python/blob/master/docs/func_groups/price_transform.md
    # AVGPRICE - Average Price
    talib_avgprice = talib.AVGPRICE(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # MEDPRICE - Median Price
    talib_medprice = talib.MEDPRICE(df.High.values, df.Low.values)
    # TYPPRICE - Typical Price
    talib_typprice = talib.TYPPRICE(
        df.High.values, df.Low.values, df.Close.values)
    # WCLPRICE - Weighted Close Price
    talib_wclprice = talib.WCLPRICE(
        df.High.values, df.Low.values, df.Close.values)

    volume_volatility_cycle_price_df = pd.DataFrame({
        # 'Date': df.Date.values,
        # 'Ticker': df.Ticker,
        # TA-Lib Volume indicators
        'ad': talib_ad,
        'adosc': talib_adosc,
        'obv': talib_obv,
        # TA-Lib Volatility indicators
        'atr': talib_atr,
        'natr': talib_natr,
        # TA-Lib Cycle Indicators
        'ht_dcperiod': talib_ht_dcperiod,
        'ht_dcphase': talib_ht_dcphase,
        'ht_phasor_inphase': talib_ht_phasor_inphase,
        'ht_phasor_quadrature': talib_ht_phasor_quadrature,
        'ht_sine_sine': talib_ht_sine_sine,
        'ht_sine_leadsine': talib_ht_sine_leadsine,
        'ht_trendmod': talib_ht_trendmode,
        # TA-Lib Price Transform Functions
        'avgprice': talib_avgprice,
        'medprice': talib_medprice,
        'typprice': talib_typprice,
        'wclprice': talib_wclprice,
        },
        index = df.index # keep the same index for pd.concat() in the next step
    )

    return volume_volatility_cycle_price_df


def talib_get_pattern_recognition_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # TA-Lib Pattern Recognition indicators
    # https://github.com/TA-Lib/ta-lib-python/blob/master/docs/func_groups/pattern_recognition.md
    # Nice article about candles (pattern recognition) https://medium.com/analytics-vidhya/recognizing-over-50-candlestick-patterns-with-python-4f02a1822cb5

    # CDL2CROWS - Two Crows
    talib_cdl2crows = talib.CDL2CROWS(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100 # normalize to [-1, 1]. [-2, 2] indicates a stronger pattern 
    # CDL3BLACKCROWS - Three Black Crows
    talib_cdl3blackrows = talib.CDL3BLACKCROWS(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDL3INSIDE - Three Inside Up/Down
    talib_cdl3inside = talib.CDL3INSIDE(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDL3LINESTRIKE - Three-Line Strike
    talib_cdl3linestrike = talib.CDL3LINESTRIKE(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDL3OUTSIDE - Three Outside Up/Down
    talib_cdl3outside = talib.CDL3OUTSIDE(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDL3STARSINSOUTH - Three Stars In The South
    talib_cdl3starsinsouth = talib.CDL3STARSINSOUTH(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDL3WHITESOLDIERS - Three Advancing White Soldiers
    talib_cdl3whitesoldiers = talib.CDL3WHITESOLDIERS(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLABANDONEDBABY - Abandoned Baby
    talib_cdlabandonedbaby = talib.CDLABANDONEDBABY(
        df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)/100
    # CDLADVANCEBLOCK - Advance Block
    talib_cdladvancedblock = talib.CDLADVANCEBLOCK(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLBELTHOLD - Belt-hold
    talib_cdlbelthold = talib.CDLBELTHOLD(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLBREAKAWAY - Breakaway
    talib_cdlbreakaway = talib.CDLBREAKAWAY(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLCLOSINGMARUBOZU - Closing Marubozu
    talib_cdlclosingmarubozu = talib.CDLCLOSINGMARUBOZU(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLCONCEALBABYSWALL - Concealing Baby Swallow
    talib_cdlconcealbabyswall = talib.CDLCONCEALBABYSWALL(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLCOUNTERATTACK - Counterattack
    talib_cdlcounterattack = talib.CDLCOUNTERATTACK(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLDARKCLOUDCOVER - Dark Cloud Cover
    talib_cdldarkcloudcover = talib.CDLDARKCLOUDCOVER(
        df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)/100
    # CDLDOJI - Doji
    talib_cdldoji = talib.CDLDOJI(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLDOJISTAR - Doji Star
    talib_cdldojistar = talib.CDLDOJISTAR(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLDRAGONFLYDOJI - Dragonfly Doji
    talib_cdldragonflydoji = talib.CDLDRAGONFLYDOJI(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLENGULFING - Engulfing Pattern
    talib_cdlengulfing = talib.CDLENGULFING(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100

    # CDLEVENINGDOJISTAR - Evening Doji Star
    talib_cdleveningdojistar = talib.CDLEVENINGDOJISTAR(
        df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)/100
    # CDLEVENINGSTAR - Evening Star
    talib_cdleveningstar = talib.CDLEVENINGSTAR(
        df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)/100
    # CDLGAPSIDESIDEWHITE - Up/Down-gap side-by-side white lines
    talib_cdlgapsidesidewhite = talib.CDLGAPSIDESIDEWHITE(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLGRAVESTONEDOJI - Gravestone Doji
    talib_cdlgravestonedoji = talib.CDLGRAVESTONEDOJI(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLHAMMER - Hammer
    talib_cdlhammer = talib.CDLHAMMER(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLHANGINGMAN - Hanging Man
    talib_cdlhangingman = talib.CDLHANGINGMAN(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLHARAMI - Harami Pattern
    talib_cdlharami = talib.CDLHARAMI(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLHARAMICROSS - Harami Cross Pattern
    talib_cdlharamicross = talib.CDLHARAMICROSS(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLHIGHWAVE - High-Wave Candle
    talib_cdlhighwave = talib.CDLHIGHWAVE(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLHIKKAKE - Hikkake Pattern
    talib_cdlhikkake = talib.CDLHIKKAKE(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLHIKKAKEMOD - Modified Hikkake Pattern
    talib_cdlhikkakemod = talib.CDLHIKKAKEMOD(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100

    # CDLHOMINGPIGEON - Homing Pigeon
    talib_cdlhomingpigeon = talib.CDLHOMINGPIGEON(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLIDENTICAL3CROWS - Identical Three Crows
    talib_cdlidentical3crows = talib.CDLIDENTICAL3CROWS(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLINNECK - In-Neck Pattern
    talib_cdlinneck = talib.CDLINNECK(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLINVERTEDHAMMER - Inverted Hammer
    talib_cdlinvertedhammer = talib.CDLINVERTEDHAMMER(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLKICKING - Kicking
    talib_cdlkicking = talib.CDLKICKING(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLKICKINGBYLENGTH - Kicking - bull/bear determined by the longer marubozu
    talib_cdlkickingbylength = talib.CDLKICKINGBYLENGTH(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLLADDERBOTTOM - Ladder Bottom
    talib_cdlladderbottom = talib.CDLLADDERBOTTOM(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLLONGLEGGEDDOJI - Long Legged Doji
    talib_cdllongleggeddoji = talib.CDLLONGLEGGEDDOJI(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLLONGLINE - Long Line Candle
    talib_cdllongline = talib.CDLLONGLINE(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLMARUBOZU - Marubozu
    talib_cdlmarubozu = talib.CDLMARUBOZU(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLMATCHINGLOW - Matching Low
    talib_cdlmatchinglow = talib.CDLMATCHINGLOW(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100

    # CDLMATHOLD - Mat Hold
    talib_cdlmathold = talib.CDLMATHOLD(
        df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)/100
    # CDLMORNINGDOJISTAR - Morning Doji Star
    talib_cdlmorningdojistar = talib.CDLMORNINGDOJISTAR(
        df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)/100
    # CDLMORNINGSTAR - Morning Star
    talib_cdlmorningstar = talib.CDLMORNINGSTAR(
        df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)/100
    # CDLONNECK - On-Neck Pattern
    talib_cdlonneck = talib.CDLONNECK(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLPIERCING - Piercing Pattern
    talib_cdlpiercing = talib.CDLPIERCING(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLRICKSHAWMAN - Rickshaw Man
    talib_cdlrickshawman = talib.CDLRICKSHAWMAN(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLRISEFALL3METHODS - Rising/Falling Three Methods
    talib_cdlrisefall3methods = talib.CDLRISEFALL3METHODS(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLSEPARATINGLINES - Separating Lines
    talib_cdlseparatinglines = talib.CDLSEPARATINGLINES(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLSHOOTINGSTAR - Shooting Star
    talib_cdlshootingstar = talib.CDLSHOOTINGSTAR(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLSHORTLINE - Short Line Candle
    talib_cdlshortline = talib.CDLSHORTLINE(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLSPINNINGTOP - Spinning Top
    talib_cdlspinningtop = talib.CDLSPINNINGTOP(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100

    # CDLSTALLEDPATTERN - Stalled Pattern
    talib_cdlstalledpattern = talib.CDLSTALLEDPATTERN(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLSTICKSANDWICH - Stick Sandwich
    talib_cdlsticksandwich = talib.CDLSTICKSANDWICH(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLTAKURI - Takuri (Dragonfly Doji with very long lower shadow)
    talib_cdltakuru = talib.CDLTAKURI(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLTASUKIGAP - Tasuki Gap
    talib_cdltasukigap = talib.CDLTASUKIGAP(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLTHRUSTING - Thrusting Pattern
    talib_cdlthrusting = talib.CDLTHRUSTING(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLTRISTAR - Tristar Pattern
    talib_cdltristar = talib.CDLTRISTAR(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLUNIQUE3RIVER - Unique 3 River
    talib_cdlunique3river = talib.CDLUNIQUE3RIVER(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLUPSIDEGAP2CROWS - Upside Gap Two Crows
    talib_cdlupsidegap2crows = talib.CDLUPSIDEGAP2CROWS(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100
    # CDLXSIDEGAP3METHODS - Upside/Downside Gap Three Methods
    talib_cdlxsidegap3methods = talib.CDLXSIDEGAP3METHODS(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)/100

    pattern_indicators_df = pd.DataFrame({
        # 'Date': df.Date.values,
        # 'Ticker': df.Ticker,
        # TA-Lib Pattern Recognition indicators
        'cdl2crows': talib_cdl2crows,
        'cdl3blackrows': talib_cdl3blackrows,
        'cdl3inside': talib_cdl3inside,
        'cdl3linestrike': talib_cdl3linestrike,
        'cdl3outside': talib_cdl3outside,
        'cdl3starsinsouth': talib_cdl3starsinsouth,
        'cdl3whitesoldiers': talib_cdl3whitesoldiers,
        'cdlabandonedbaby': talib_cdlabandonedbaby,
        'cdladvancedblock': talib_cdladvancedblock,
        'cdlbelthold': talib_cdlbelthold,
        'cdlbreakaway': talib_cdlbreakaway,
        'cdlclosingmarubozu': talib_cdlclosingmarubozu,
        'cdlconcealbabyswall': talib_cdlconcealbabyswall,
        'cdlcounterattack': talib_cdlcounterattack,
        'cdldarkcloudcover': talib_cdldarkcloudcover,
        'cdldoji': talib_cdldoji,
        'cdldojistar': talib_cdldojistar,
        'cdldragonflydoji': talib_cdldragonflydoji,
        'cdlengulfing': talib_cdlengulfing,
        'cdleveningdojistar': talib_cdleveningdojistar,
        'cdleveningstar': talib_cdleveningstar,
        'cdlgapsidesidewhite': talib_cdlgapsidesidewhite,
        'cdlgravestonedoji': talib_cdlgravestonedoji,
        'cdlhammer': talib_cdlhammer,
        'cdlhangingman': talib_cdlhangingman,
        'cdlharami': talib_cdlharami,
        'cdlharamicross': talib_cdlharamicross,
        'cdlhighwave': talib_cdlhighwave,
        'cdlhikkake': talib_cdlhikkake,
        'cdlhikkakemod': talib_cdlhikkakemod,
        'cdlhomingpigeon': talib_cdlhomingpigeon,
        'cdlidentical3crows': talib_cdlidentical3crows,
        'cdlinneck': talib_cdlinneck,
        'cdlinvertedhammer': talib_cdlinvertedhammer,
        'cdlkicking': talib_cdlkicking,
        'cdlkickingbylength': talib_cdlkickingbylength,
        'cdlladderbottom': talib_cdlladderbottom,
        'cdllongleggeddoji': talib_cdllongleggeddoji,
        'cdllongline': talib_cdllongline,
        'cdlmarubozu': talib_cdlmarubozu,
        'cdlmatchinglow': talib_cdlmatchinglow,
        'cdlmathold': talib_cdlmathold,
        'cdlmorningdojistar': talib_cdlmorningdojistar,
        'cdlmorningstar': talib_cdlmorningstar,
        'cdlonneck': talib_cdlonneck,
        'cdlpiercing': talib_cdlpiercing,
        'cdlrickshawman': talib_cdlrickshawman,
        'cdlrisefall3methods': talib_cdlrisefall3methods,
        'cdlseparatinglines': talib_cdlseparatinglines,
        'cdlshootingstar': talib_cdlshootingstar,
        'cdlshortline': talib_cdlshortline,
        'cdlspinningtop': talib_cdlspinningtop,
        'cdlstalledpattern': talib_cdlstalledpattern,
        'cdlsticksandwich': talib_cdlsticksandwich,
        'cdltakuru': talib_cdltakuru,
        'cdltasukigap': talib_cdltasukigap,
        'cdlthrusting': talib_cdlthrusting,
        'cdltristar': talib_cdltristar,
        'cdlunique3river': talib_cdlunique3river,
        'cdlupsidegap2crows': talib_cdlupsidegap2crows,
        'cdlxsidegap3methods': talib_cdlxsidegap3methods
        },
        index = df.index # keep the same index for pd.concat() in the next step
    )

    return pattern_indicators_df

def get_tech_indices(stocks_df):
    ALL_TICKERS = stocks_df["Ticker"].unique().tolist()

    # adding Momentum / Pattern/ Volume features to all tickers - one by one
    results = []

    current_ticker_data = None
    i=0
    for ticker in ALL_TICKERS: # MUST do it one by one!! ZB20251105
        i+=1
        print(f'{i}/{len(ALL_TICKERS)} Current ticker is {ticker}')
        current_ticker_data = stocks_df[stocks_df.Ticker == ticker]
    
        # 3 calls to get additional features
        df_current_ticker_momentum_indicators = talib_get_momentum_indicators_for_one_ticker(current_ticker_data)  
        df_current_ticker_volume_indicators = talib_get_volume_volatility_cycle_price_indicators(current_ticker_data)
        df_current_ticker_pattern_indicators = talib_get_pattern_recognition_indicators(current_ticker_data)
    
        # merge the above 3 results
        current_ticker_data_merged = pd.concat([current_ticker_data, # all dfs have the same index to expand columns
                                                df_current_ticker_momentum_indicators,
                                                df_current_ticker_volume_indicators,
                                                df_current_ticker_pattern_indicators],
                                               axis = 1
                                              )
    
        # add to results list for concating
        results.append(current_ticker_data_merged)
        
    # merge to the output df
    merged_df_with_tech_ind = pd.concat(results, # a list of 3 dfs
                                        axis = 0
                                       )
   
    return merged_df_with_tech_ind


def save_data(df):
    """Save data to parquet files"""
   
    data_dir = os.path.join(os.getcwd(), "data")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    today = date.today().strftime("%Y%m%d")
    
    if not df.empty:
        file_path = os.path.join(data_dir, f"{today}_df_wTarget_wAllFeatures.parquet")
        try:
            df.to_parquet(file_path, compression="brotli")
            print(f"df_tar Saved to {file_path}")
        except Exception as e:
            print(f"Error saving file: {e}")

    else:
        print(f"Warning: df is empty, not saving")

def main():
    print("loading data")
    stocks_df = load_data()
    
    print("preparing data")
    stocks_df = prepare_data(stocks_df)

    print("getting custom features")
    stocks_df = get_custom_features(stocks_df)
    
    print("getting tech indicators")
    stocks_with_tech_indicators_df = get_tech_indices(stocks_df)
    
    print("saving data")
    save_data(stocks_with_tech_indicators_df)
    
    print("Data transformation completed!")
    return  stocks_with_tech_indicators_df

if __name__ == "__main__":
    stocks_with_tech_indicators_df = main()
