#!/usr/bin/env python
# coding: utf-8

# IMPORTS
import numpy as np
import pandas as pd
import os
import joblib

# Fin Data Sources
import yfinance as yf
import pandas_datareader as pdr

# Data viz
import plotly.graph_objs as go
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Measure time for ML HyperParams search
import time
from datetime import date

# ML models and utils
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score

# Disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None

# Load data
data_dir = os.getcwd() + "/data"
file_name = "20251115_df_wTarget_wAllFeatures.parquet"
print("About to load parquet file...")
df = pd.read_parquet(os.path.join(data_dir, file_name))
print(f"Loaded successfully: {df.shape}")

# convert +-inf to NaN, and then drop them, along with existing NaN.
def clean_dataframe_from_inf_and_nan(df:pd.DataFrame):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True) # drop nan not fillna, to better mimic the market
    return df

# Clean datasets
print("Starting data cleaning...")
# No more NaN will be created after this point. remove NaN before splitting.
df = clean_dataframe_from_inf_and_nan(df)
print("datasets cleaned")


# Define feature groups
GROWTH = [g for g in df.keys() if (g.find('growth_')==0)&(g.find('future')<0)]
OHLCV = ['Open','High','Low','Close','Volume']
CATEGORICAL = ['Month', 'Weekday', 'Ticker']

# Define prediction targets
TO_PREDICT = [g for g in df.keys() if (g.find('future')>=0)]

# customized moving averages
MAs = [g for g in df.keys() if (g.find('MA')>=0)]

# Define features to drop
TO_DROP = ['Year','Date', 'returns'] + CATEGORICAL + OHLCV

# Add a small constant to avoid log(0)
df['ln_volume'] = df.Volume.apply(lambda x: np.log(x+ 1e-6))

# manually defined features
CUSTOM_NUMERICAL = MAs + ['growing_moving_average', 'high_minus_low_relative','volatility', 'ln_volume']

# Ta-lib indicators
TECHNICAL_INDICATORS = ['adx', 'adxr', 'apo', 'aroon_1','aroon_2', 'aroonosc',
 'bop', 'cci', 'cmo','dx', 'macd', 'macdsignal', 'macdhist', 'macd_ext',
 'macdsignal_ext', 'macdhist_ext', 'macd_fix', 'macdsignal_fix',
 'macdhist_fix', 'mfi', 'minus_di', 'mom', 'plus_di', 'dm', 'ppo',
 'roc', 'rocp', 'rocr', 'rocr100', 'rsi', 'slowk', 'slowd', 'fastk',
 'fastd', 'fastk_rsi', 'fastd_rsi', 'trix', 'ultosc', 'willr',
 'ad', 'adosc', 'obv', 'atr', 'natr', 'ht_dcperiod', 'ht_dcphase',
 'ht_phasor_inphase', 'ht_phasor_quadrature', 'ht_sine_sine', 'ht_sine_leadsine',
 'ht_trendmod', 'avgprice', 'medprice', 'typprice', 'wclprice']

# Ta-lib patterns
TECHNICAL_PATTERNS = [g for g in df.keys() if g.find('cdl')>=0]

# Merge all numerical numbers
NUMERICAL = GROWTH + CUSTOM_NUMERICAL + TECHNICAL_INDICATORS + TECHNICAL_PATTERNS

# Create dummy variables
df['Month'] = df.Month.astype(str)    
df['Weekday'] = df.Weekday.astype(str) 
print("Month and Weekday columns created successfully")

# Generate dummy variables (no need for bool, let's have int32 instead)
dummy_variables = pd.get_dummies(df[CATEGORICAL], dtype='int32')
DUMMIES = dummy_variables.keys().to_list()
print(f"Created {len(DUMMIES)} dummy variables")

# Concatenate dummy variables with original DataFrame
df_with_dummies = pd.concat([df, dummy_variables], axis=1)
print("Concatenation completed")
print(f"df_with_dummies shape: {df_with_dummies.shape}")
print(f"Memory usage: {df_with_dummies.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

def temporal_split(df, min_date, max_date, train_prop=0.7, val_prop=0.15, test_prop=0.15):
    """
    Splits a DataFrame into three buckets based on the temporal order of the 'Date' column.
    """
    train_end = min_date + pd.Timedelta(days=(max_date - min_date).days * train_prop)
    val_end = train_end + pd.Timedelta(days=(max_date - min_date).days * val_prop)

    split_labels = []
    for date in df['Date']:
        if date <= train_end:
            split_labels.append('train')
        elif date <= val_end:
            split_labels.append('validation')
        else:
            split_labels.append('test')

    df['split'] = split_labels
    return df

print("Getting min/max dates...")
min_date_df = df_with_dummies.Date.min()
max_date_df = df_with_dummies.Date.max()
print(f"Date range: {min_date_df} to {max_date_df}")

# Apply temporal split
min_date_df = df_with_dummies.Date.min()
max_date_df = df_with_dummies.Date.max()
print("Starting temporal split...")
df_with_dummies = temporal_split(df_with_dummies, min_date=min_date_df, max_date=max_date_df)
print("Temporal split completed")
# Create clean copy
print("Creating clean dataframe copy...")
new_df = df_with_dummies.copy()
print("Clean copy completed")

# Prepare feature sets
print("Preparing feature sets...")
features_list = NUMERICAL + DUMMIES
print(f"Features list length: {len(features_list)}")

# set up the target
to_predict = 'is_positive_growth_24h_future'
print("We're trying to predict:", to_predict)

# Create data splits
print("Creating train split...")
train_df = new_df[new_df.split.isin(['train'])].copy(deep=True)
print("Creating valid split...")
valid_df = new_df[new_df.split.isin(['validation'])].copy(deep=True)
print("Creating train_valid split...")
train_valid_df = new_df[new_df.split.isin(['train','validation'])].copy(deep=True)
print("Creating test split...")
test_df = new_df[new_df.split.isin(['test'])].copy(deep=True)
print("Data splits completed")

# Prepare feature matrices
print("Creating feature matrices...")
print(f"About to create X_train with {len(features_list+[to_predict])} columns...")
X_train = train_df[features_list+[to_predict]]
print("X_train created")
print("Creating X_valid...")
X_valid = valid_df[features_list+[to_predict]]
print("X_valid created")
print("Creating X_train_valid...")
X_test = test_df[features_list+[to_predict]]
print("X_train_valid created")
print("Creating X_test...")
X_train_valid = train_valid_df[features_list+[to_predict]]
print("X_test created")
print("Creating X_all...")
X_all = new_df[features_list+[to_predict]].copy(deep=True)
print("X_all created")

# Separate targets from features
y_train = X_train[to_predict]
y_valid = X_valid[to_predict]
y_train_valid = X_train_valid[to_predict]
y_test = X_test[to_predict]
y_all = X_all[to_predict]

del X_train[to_predict]
del X_valid[to_predict]
del X_train_valid[to_predict]
del X_test[to_predict]
del X_all[to_predict]

# Modeling
# Generate manual predictions
new_df['pred0_manual_cci'] = (new_df.cci>200).astype(int)
new_df['pred1_manual_prev_g1'] = (new_df.his_growth_24h>1).astype(int)
new_df['pred2_manual_prev_g1_and_sma50'] = ((new_df['his_growth_24h'] > 1) & (new_df['low_touching_SMA50_relative'] < -0.04)).astype(int)
new_df['pred3_manual_prev_g1_and_sma200'] = ((new_df['his_growth_24h'] > 1) & (new_df['low_touching_SMA200_relative'] < -0.05)).astype(int)

def get_predictions_correctness(df: pd.DataFrame, to_predict: str):
    """Calculate prediction correctness and precision on test set"""
    PREDICTIONS = [k for k in df.keys() if k.startswith('pred')]

    # Add correctness columns
    for pred in PREDICTIONS:
        part1 = pred.split('_')[0]
        df[f'is_correct_{part1}'] = (df[pred] == df[to_predict]).astype(int)

    IS_CORRECT = [k for k in df.keys() if k.startswith('is_correct_')]
    return PREDICTIONS, IS_CORRECT

PREDICTIONS, IS_CORRECT = get_predictions_correctness(df=new_df, to_predict=to_predict)

# Decision Tree Classifier functions
def fit_decision_tree(X, y, max_depth=20):
    """Fit decision tree classifier"""
    print(f"INSIDE fit_decision_tree function with max_depth={max_depth}")
    print(f"  -> Fitting Decision Tree with max_depth={max_depth} on {X.shape[0]} samples...")
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X, y)
    return clf, X.columns

print(f"About to train with X_train_valid shape: {X_train_valid.shape}")
print(f"y_train_valid shape: {y_train_valid.shape}")
print(f"X_train_valid dtypes: {X_train_valid.dtypes.value_counts()}")
print("Calling fit_decision_tree now...")
# Train decision trees with different depths
clf_13, train_columns = fit_decision_tree(X=X_train_valid, y=y_train_valid, max_depth=13)
y_pred_all = clf_13.predict(X_all)
new_df['pred4_clf_13'] = y_pred_all




# Best hyperparameters from previous tuning
precision_by_depth = {1: 0.6349, 2: 0.6428, 3: 0.6164, 4: 0.6304, 5: 0.6224, 6: 0.6217, 7: 0.6197, 8: 0.6247, 9: 0.6262, 10: 0.6227, 11: 0.6124, 12: 0.6156, 13: 0.6115, 14: 0.6066, 15: 0.6122, 16: 0.6052, 17: 0.5989, 18: 0.5955, 19: 0.6021, 20: 0.605}

clf_4, train_columns = fit_decision_tree(X=X_train_valid, y=y_train_valid, max_depth=4)
y_pred_all = clf_4.predict(X_all)
new_df['pred6_clf_4'] = y_pred_all

# Best depth model
best_depth = 2
best_precision = precision_by_depth[best_depth]

clf_best, train_columns = fit_decision_tree(X=X_train_valid, y=y_train_valid, max_depth=best_depth)
y_pred_clf_best = clf_best.predict(X_all)
new_df['pred6_clf_best'] = y_pred_clf_best



# Random Forest
# Pre-calculated best precision matrix
best_precision_matrix_random_forest = {(5, 10): 0.6147, (5, 50): 0.6061, (5, 100): 0.6081, (5, 200): 0.6089, (7, 10): 0.6171, (7, 50): 0.614, (7, 100): 0.6178, (7, 200): 0.6191, (9, 10): 0.6151, (9, 50): 0.6239, (9, 100): 0.6246, (9, 200): 0.6247, (11, 10): 0.6154, (11, 50): 0.6241, (11, 100): 0.6263, (11, 200): 0.628, (13, 10): 0.6217, (13, 50): 0.6256, (13, 100): 0.6264, (13, 200): 0.6277, (15, 10): 0.6197, (15, 50): 0.6241, (15, 100): 0.6251, (15, 200): 0.6269, (17, 10): 0.6188, (17, 50): 0.6253, (17, 100): 0.6269, (17, 200): 0.6284, (19, 10): 0.6217, (19, 50): 0.6268, (19, 100): 0.6277, (19, 200): 0.6279, (21, 10): 0.6198, (21, 50): 0.6289, (21, 100): 0.6269, (21, 200): 0.6262}

# Best Random Forest hyperparameters
rf_best_n_estimators = 200
rf_best_max_depth = 17

# Train Random Forest
rf_best = RandomForestClassifier(n_estimators=rf_best_n_estimators, max_depth=rf_best_max_depth, random_state=42, n_jobs=-1)
rf_best = rf_best.fit(X_train_valid, y_train_valid)


# XGBoost
precision_matrix = {(2, 50): 0.6205, (2, 100): 0.6287, (2, 200): 0.6373, (3, 50): 0.6275, (3, 100): 0.6341, (3, 200): 0.6369, (4, 50): 0.6304, (4, 100): 0.6397, (4, 200): 0.6397, (5, 50): 0.6331, (5, 100): 0.6382, (5, 200): 0.6393, (7, 50): 0.6342, (7, 100): 0.6337, (7, 200): 0.6365}
# Best XGBoost hyperparameters
xgc_best_n_estimators = 200
xgc_best_max_depth = 4 

xgc_best = XGBClassifier(objective='binary:logistic', 
                        n_estimators=xgc_best_n_estimators, 
                        learning_rate=0.1, 
                        max_depth=xgc_best_max_depth, 
                        random_state=42)
xgc_best = xgc_best.fit(X_train_valid, y_train_valid)


def tpr_fpr_dataframe(y_true, y_pred, only_even=False):
    """Calculate TPR/FPR for different thresholds"""
    scores = []

    if only_even == False:
        thresholds = np.linspace(0, 1, 101)
    else:
        thresholds = np.linspace(0, 1, 51)

    for t in thresholds:
        actual_positive = (y_true == 1)
        actual_negative = (y_true == 0)
        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)

        tp = (predict_positive & actual_positive).sum()
        tn = (predict_negative & actual_negative).sum()
        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()

        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0

        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0

        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0

        accuracy = (tp + tn) / (tp + tn + fp + fn)

        scores.append((t, tp, fp, fn, tn, precision, recall, accuracy, f1_score))

    columns = ['threshold', 'tp', 'fp', 'fn', 'tn', 'precision', 'recall', 'accuracy', 'f1_score']
    df_scores = pd.DataFrame(scores, columns=columns)

    df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
    df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)

    return df_scores

# Generate threshold-based predictions for Decision Tree
y_pred_all = clf_best.predict_proba(X_all)
y_pred_all_class1 = [k[1] for k in y_pred_all]
y_pred_all_class1_array = np.array(y_pred_all_class1)

new_df['proba_pred8'] = y_pred_all_class1_array
new_df['pred8_clf_second_best_rule_84'] = (y_pred_all_class1_array >= 0.84).astype(int)
new_df['proba_pred9'] = y_pred_all_class1_array
new_df['pred9_clf_second_best_rule_92'] = (y_pred_all_class1_array >= 0.92).astype(int)

# Generate threshold-based predictions for Random Forest
y_pred_all = rf_best.predict_proba(X_all)
y_pred_all_class1 = [k[1] for k in y_pred_all]
y_pred_all_class1_array = np.array(y_pred_all_class1)

new_df['proba_pred10'] = y_pred_all_class1_array
new_df['pred10_rf_best_rule_55'] = (y_pred_all_class1_array >= 0.55).astype(int)
new_df['proba_pred11'] = y_pred_all_class1_array
new_df['pred11_rf_best_rule_65'] = (y_pred_all_class1_array >= 0.65).astype(int)

# Generate threshold-based predictions for XGBoost
y_pred_all = xgc_best.predict_proba(X_all)
y_pred_all_class1 = [k[1] for k in y_pred_all] #list of predictions for class "1"
y_pred_all_class1_array = np.array(y_pred_all_class1) # (Numpy Array) np.array of predictions for class "1" , converted from a list

# adding XGBoost predictors (xgc_best) to the dataset for 2 new rules, based on Thresholds
# defining a new prediction vector is easy now, as the dimensions will match
new_df['proba_pred12'] = y_pred_all_class1_array
new_df['pred12_xgc_best_rule_55'] = (y_pred_all_class1_array >= 0.55).astype(int)
new_df['proba_pred13'] = y_pred_all_class1_array
new_df['pred13_xgc_best_rule_65'] = (y_pred_all_class1_array >= 0.65).astype(int)


# Final update of predictions and correctness
print("Final update of predictions and correctness...")
PREDICTIONS, IS_CORRECT = get_predictions_correctness(new_df, to_predict=to_predict)
print(f"Total predictions generated: {len(PREDICTIONS)}")

# Save the final dataframe
print("Saving final dataframe...")
filename = "new_df_wProbability_entrySignals.joblib"
path = os.path.join(data_dir, filename)
joblib.dump(new_df, path)
print(f"Dataframe saved to: {path}")
print("Model preparation completed successfully!")
