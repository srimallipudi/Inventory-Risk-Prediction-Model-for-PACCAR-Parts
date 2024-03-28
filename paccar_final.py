#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 10:02:46 2023

@author: srilu
"""

# Importing the relevant libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict

# Loading the training and testing data
dfi = pd.read_csv('training_data_0101_0514.csv')
dfi_test = pd.read_csv('test_data_0727_1031.csv')
print(dfi)
print(dfi.shape) 
print(dfi.nunique()) 
print(dfi.isnull().sum().sum()) # Total Null value counts
print(dfi.isnull().sum())

# Encoding Categorical data
cat_cols = ['expid', 'pdc', 'desk']
le = LabelEncoder()

for col in cat_cols:
    dfi[col] = le.fit_transform(dfi[col])
    dfi_test[col] = le.fit_transform(dfi_test[col])

# Encoding Suggestion date column to months
dfi['suggestion_dt'] = pd.to_datetime(dfi['suggestion_dt'])
dfi['suggestion_dt'] = dfi['suggestion_dt'].dt.month

dfi_test['suggestion_dt'] = pd.to_datetime(dfi_test['suggestion_dt'])
dfi_test['suggestion_dt'] = dfi_test['suggestion_dt'].dt.month

# Define dictionary to map velocity labels to numerical values
velocity = {'A': 10, 'E': 11, 'L': 12, 'M': 13, 'N': 14, 'S': 15, 'T': 16, 'D': 17}
dfi['velocity'] = dfi['velocity'].replace(velocity)
dfi_test['velocity'] = dfi_test['velocity'].replace(velocity)

# Filling the missing values
dfi["part_cost"] = dfi.groupby("item_id")["part_cost"].transform(lambda x: x.fillna(x.mean()))
dfi["part_cost"] = dfi["part_cost"].transform(lambda x: x.fillna(x.mean()))
dfi_test["part_cost"] = dfi_test.groupby("item_id")["part_cost"].transform(lambda x: x.fillna(x.mean()))
dfi_test["part_cost"] = dfi_test["part_cost"].transform(lambda x: x.fillna(x.mean()))


dfi["ss_units_left_pct"] = dfi.groupby(["item_id", "pdc"])["ss_units_left_pct"].transform(lambda x: x.fillna(x.median()))
dfi["ss_units_left_pct"] = dfi.groupby(["item_id"])["ss_units_left_pct"].transform(lambda x: x.fillna(x.median()))
dfi["ss_units_left_pct"] = dfi["ss_units_left_pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["ss_units_left_pct"] = dfi_test.groupby(["item_id", "pdc"])["ss_units_left_pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["ss_units_left_pct"] = dfi_test.groupby(["item_id"])["ss_units_left_pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["ss_units_left_pct"] = dfi_test["ss_units_left_pct"].transform(lambda x: x.fillna(x.median()))


dfi["max_oh_left_pct"] = dfi.groupby(["item_id", "pdc"])["max_oh_left_pct"].transform(lambda x: x.fillna(x.median()))
dfi["max_oh_left_pct"] = dfi.groupby(["item_id"])["max_oh_left_pct"].transform(lambda x: x.fillna(x.median()))
dfi["max_oh_left_pct"] = dfi["max_oh_left_pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["max_oh_left_pct"] = dfi_test.groupby(["item_id", "pdc"])["max_oh_left_pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["max_oh_left_pct"] = dfi_test.groupby(["item_id"])["max_oh_left_pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["max_oh_left_pct"] = dfi_test["max_oh_left_pct"].transform(lambda x: x.fillna(x.median()))


dfi["oh_5d_change"] = dfi.groupby(["item_id", "pdc"])["oh_5d_change"].transform(lambda x: x.fillna(x.median()))
dfi["oh_5d_change"] = dfi.groupby(["item_id"])["oh_5d_change"].transform(lambda x: x.fillna(x.median()))
dfi["oh_5d_change"] = dfi["oh_5d_change"].transform(lambda x: x.fillna(x.median()))
dfi_test["oh_5d_change"] = dfi_test.groupby(["item_id", "pdc"])["oh_5d_change"].transform(lambda x: x.fillna(x.median()))
dfi_test["oh_5d_change"] = dfi_test.groupby(["item_id"])["oh_5d_change"].transform(lambda x: x.fillna(x.median()))
dfi_test["oh_5d_change"] = dfi_test["oh_5d_change"].transform(lambda x: x.fillna(x.median()))


dfi["min_on_hand_change_5d"] = dfi.groupby(["item_id", "pdc"])["min_on_hand_change_5d"].transform(lambda x: x.fillna(x.median()))
dfi["min_on_hand_change_5d"] = dfi.groupby(["item_id"])["min_on_hand_change_5d"].transform(lambda x: x.fillna(x.median()))
dfi["min_on_hand_change_5d"] = dfi["min_on_hand_change_5d"].transform(lambda x: x.fillna(x.median()))
dfi_test["min_on_hand_change_5d"] = dfi_test.groupby(["item_id", "pdc"])["min_on_hand_change_5d"].transform(lambda x: x.fillna(x.median()))
dfi_test["min_on_hand_change_5d"] = dfi_test.groupby(["item_id"])["min_on_hand_change_5d"].transform(lambda x: x.fillna(x.median()))
dfi_test["min_on_hand_change_5d"] = dfi_test["min_on_hand_change_5d"].transform(lambda x: x.fillna(x.median()))


dfi["supplier_past_due_pct"] = dfi.groupby(["vndr_concat", "item_id", "pdc"])["supplier_past_due_pct"].transform(lambda x: x.fillna(x.median()))
dfi["supplier_past_due_pct"] = dfi.groupby(["vndr_concat", "item_id"])["supplier_past_due_pct"].transform(lambda x: x.fillna(x.median()))
dfi["supplier_past_due_pct"] = dfi.groupby(["vndr_concat"])["supplier_past_due_pct"].transform(lambda x: x.fillna(x.median()))
dfi["supplier_past_due_pct"] = dfi.groupby(["item_id"])["supplier_past_due_pct"].transform(lambda x: x.fillna(x.median()))
dfi["supplier_past_due_pct"] = dfi["supplier_past_due_pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["supplier_past_due_pct"] = dfi_test.groupby(["vndr_concat", "item_id", "pdc"])["supplier_past_due_pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["supplier_past_due_pct"] = dfi_test.groupby(["vndr_concat", "item_id"])["supplier_past_due_pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["supplier_past_due_pct"] = dfi_test.groupby(["vndr_concat"])["supplier_past_due_pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["supplier_past_due_pct"] = dfi_test.groupby(["item_id"])["supplier_past_due_pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["supplier_past_due_pct"] = dfi_test["supplier_past_due_pct"].transform(lambda x: x.fillna(x.median()))


dfi["ots_pct"] = dfi.groupby(["vndr_concat", "item_id", "pdc"])["ots_pct"].transform(lambda x: x.fillna(x.median()))
dfi["ots_pct"] = dfi.groupby(["vndr_concat", "item_id"])["ots_pct"].transform(lambda x: x.fillna(x.median()))
dfi["ots_pct"] = dfi.groupby(["vndr_concat"])["ots_pct"].transform(lambda x: x.fillna(x.median()))
dfi["ots_pct"] = dfi.groupby(["item_id"])["ots_pct"].transform(lambda x: x.fillna(x.median()))
dfi["ots_pct"] = dfi["ots_pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["ots_pct"] = dfi_test.groupby(["vndr_concat", "item_id", "pdc"])["ots_pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["ots_pct"] = dfi_test.groupby(["vndr_concat", "item_id"])["ots_pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["ots_pct"] = dfi_test.groupby(["vndr_concat"])["ots_pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["ots_pct"] = dfi_test.groupby(["item_id"])["ots_pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["ots_pct"] = dfi_test["ots_pct"].transform(lambda x: x.fillna(x.median()))


dfi["early_ratio"] = dfi.groupby(["vndr_concat", "item_id", "pdc"])["early_ratio"].transform(lambda x: x.fillna(x.median()))
dfi["early_ratio"] = dfi.groupby(["vndr_concat", "item_id"])["early_ratio"].transform(lambda x: x.fillna(x.median()))
dfi["early_ratio"] = dfi.groupby(["vndr_concat"])["early_ratio"].transform(lambda x: x.fillna(x.median()))
dfi["early_ratio"] = dfi.groupby(["item_id"])["early_ratio"].transform(lambda x: x.fillna(x.median()))
dfi["early_ratio"] = dfi["early_ratio"].transform(lambda x: x.fillna(x.median()))
dfi_test["early_ratio"] = dfi_test.groupby(["vndr_concat", "item_id", "pdc"])["early_ratio"].transform(lambda x: x.fillna(x.median()))
dfi_test["early_ratio"] = dfi_test.groupby(["vndr_concat", "item_id"])["early_ratio"].transform(lambda x: x.fillna(x.median()))
dfi_test["early_ratio"] = dfi_test.groupby(["vndr_concat"])["early_ratio"].transform(lambda x: x.fillna(x.median()))
dfi_test["early_ratio"] = dfi_test.groupby(["item_id"])["early_ratio"].transform(lambda x: x.fillna(x.median()))
dfi_test["early_ratio"] = dfi_test["early_ratio"].transform(lambda x: x.fillna(x.median()))


dfi["on_time_ratio"] = dfi.groupby(["vndr_concat", "item_id", "pdc"])["on_time_ratio"].transform(lambda x: x.fillna(x.median()))
dfi["on_time_ratio"] = dfi.groupby(["vndr_concat", "item_id"])["on_time_ratio"].transform(lambda x: x.fillna(x.median()))
dfi["on_time_ratio"] = dfi.groupby(["vndr_concat"])["on_time_ratio"].transform(lambda x: x.fillna(x.median()))
dfi["on_time_ratio"] = dfi.groupby(["item_id"])["on_time_ratio"].transform(lambda x: x.fillna(x.median()))
dfi["on_time_ratio"] = dfi["on_time_ratio"].transform(lambda x: x.fillna(x.median()))
dfi_test["on_time_ratio"] = dfi_test.groupby(["vndr_concat", "item_id", "pdc"])["on_time_ratio"].transform(lambda x: x.fillna(x.median()))
dfi_test["on_time_ratio"] = dfi_test.groupby(["vndr_concat", "item_id"])["on_time_ratio"].transform(lambda x: x.fillna(x.median()))
dfi_test["on_time_ratio"] = dfi_test.groupby(["vndr_concat"])["on_time_ratio"].transform(lambda x: x.fillna(x.median()))
dfi_test["on_time_ratio"] = dfi_test.groupby(["item_id"])["on_time_ratio"].transform(lambda x: x.fillna(x.median()))
dfi_test["on_time_ratio"] = dfi_test["on_time_ratio"].transform(lambda x: x.fillna(x.median()))


dfi["no_ship_ratio"] = dfi.groupby(["vndr_concat", "item_id", "pdc"])["no_ship_ratio"].transform(lambda x: x.fillna(x.median()))
dfi["no_ship_ratio"] = dfi.groupby(["vndr_concat", "item_id"])["no_ship_ratio"].transform(lambda x: x.fillna(x.median()))
dfi["no_ship_ratio"] = dfi.groupby(["vndr_concat"])["no_ship_ratio"].transform(lambda x: x.fillna(x.median()))
dfi["no_ship_ratio"] = dfi.groupby(["item_id"])["no_ship_ratio"].transform(lambda x: x.fillna(x.median()))
dfi["no_ship_ratio"] = dfi["no_ship_ratio"].transform(lambda x: x.fillna(x.median()))
dfi_test["no_ship_ratio"] = dfi_test.groupby(["vndr_concat", "item_id", "pdc"])["no_ship_ratio"].transform(lambda x: x.fillna(x.median()))
dfi_test["no_ship_ratio"] = dfi_test.groupby(["vndr_concat", "item_id"])["no_ship_ratio"].transform(lambda x: x.fillna(x.median()))
dfi_test["no_ship_ratio"] = dfi_test.groupby(["vndr_concat"])["no_ship_ratio"].transform(lambda x: x.fillna(x.median()))
dfi_test["no_ship_ratio"] = dfi_test.groupby(["item_id"])["no_ship_ratio"].transform(lambda x: x.fillna(x.median()))
dfi_test["no_ship_ratio"] = dfi_test["no_ship_ratio"].transform(lambda x: x.fillna(x.median()))


dfi["dmd_rolling_90d"] = dfi.groupby(["item_id", "pdc"])["dmd_rolling_90d"].transform(lambda x: x.fillna(x.median()))
dfi["dmd_rolling_90d"] = dfi.groupby(["item_id"])["dmd_rolling_90d"].transform(lambda x: x.fillna(x.median()))
dfi["dmd_rolling_90d"] = dfi["dmd_rolling_90d"].transform(lambda x: x.fillna(x.median()))
dfi_test["dmd_rolling_90d"] = dfi_test.groupby(["item_id", "pdc"])["dmd_rolling_90d"].transform(lambda x: x.fillna(x.median()))
dfi_test["dmd_rolling_90d"] = dfi_test.groupby(["item_id"])["dmd_rolling_90d"].transform(lambda x: x.fillna(x.median()))
dfi_test["dmd_rolling_90d"] = dfi_test["dmd_rolling_90d"].transform(lambda x: x.fillna(x.median()))


dfi["dmd_fcst_portion"] = dfi.groupby(["item_id", "pdc"])["dmd_fcst_portion"].transform(lambda x: x.fillna(x.median()))
dfi["dmd_fcst_portion"] = dfi.groupby(["item_id"])["dmd_fcst_portion"].transform(lambda x: x.fillna(x.median()))
dfi["dmd_fcst_portion"] = dfi["dmd_fcst_portion"].transform(lambda x: x.fillna(x.median()))
dfi_test["dmd_fcst_portion"] = dfi_test.groupby(["item_id", "pdc"])["dmd_fcst_portion"].transform(lambda x: x.fillna(x.median()))
dfi_test["dmd_fcst_portion"] = dfi_test.groupby(["item_id"])["dmd_fcst_portion"].transform(lambda x: x.fillna(x.median()))
dfi_test["dmd_fcst_portion"] = dfi_test["dmd_fcst_portion"].transform(lambda x: x.fillna(x.median()))


dfi["orders_12m"] = dfi.groupby(["vndr_concat", "item_id", "pdc"])["orders_12m"].transform(lambda x: x.fillna(x.median()))
dfi["orders_12m"] = dfi.groupby(["vndr_concat", "item_id"])["orders_12m"].transform(lambda x: x.fillna(x.median()))
dfi["orders_12m"] = dfi.groupby(["vndr_concat"])["orders_12m"].transform(lambda x: x.fillna(x.median()))
dfi["orders_12m"] = dfi.groupby(["item_id"])["orders_12m"].transform(lambda x: x.fillna(x.median()))
dfi["orders_12m"] = dfi["orders_12m"].transform(lambda x: x.fillna(x.median()))
dfi_test["orders_12m"] = dfi_test.groupby(["vndr_concat", "item_id", "pdc"])["orders_12m"].transform(lambda x: x.fillna(x.median()))
dfi_test["orders_12m"] = dfi_test.groupby(["vndr_concat", "item_id"])["orders_12m"].transform(lambda x: x.fillna(x.median()))
dfi_test["orders_12m"] = dfi_test.groupby(["vndr_concat"])["orders_12m"].transform(lambda x: x.fillna(x.median()))
dfi_test["orders_12m"] = dfi_test.groupby(["item_id"])["orders_12m"].transform(lambda x: x.fillna(x.median()))
dfi_test["orders_12m"] = dfi_test["orders_12m"].transform(lambda x: x.fillna(x.median()))


dfi["dmd_wkly_95pct"] = dfi.groupby(["item_id", "pdc"])["dmd_wkly_95pct"].transform(lambda x: x.fillna(x.median()))
dfi["dmd_wkly_95pct"] = dfi.groupby(["item_id"])["dmd_wkly_95pct"].transform(lambda x: x.fillna(x.median()))
dfi["dmd_wkly_95pct"] = dfi["dmd_wkly_95pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["dmd_wkly_95pct"] = dfi_test.groupby(["item_id", "pdc"])["dmd_wkly_95pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["dmd_wkly_95pct"] = dfi_test.groupby(["item_id"])["dmd_wkly_95pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["dmd_wkly_95pct"] = dfi_test["dmd_wkly_95pct"].transform(lambda x: x.fillna(x.median()))


dfi["dmd_wkly_dos"] = dfi.groupby(["item_id", "pdc"])["dmd_wkly_dos"].transform(lambda x: x.fillna(x.median()))
dfi["dmd_wkly_dos"] = dfi.groupby(["item_id"])["dmd_wkly_dos"].transform(lambda x: x.fillna(x.median()))
dfi["dmd_wkly_dos"] = dfi["dmd_wkly_dos"].transform(lambda x: x.fillna(x.median()))
dfi_test["dmd_wkly_dos"] = dfi_test.groupby(["item_id", "pdc"])["dmd_wkly_dos"].transform(lambda x: x.fillna(x.median()))
dfi_test["dmd_wkly_dos"] = dfi_test.groupby(["item_id"])["dmd_wkly_dos"].transform(lambda x: x.fillna(x.median()))
dfi_test["dmd_wkly_dos"] = dfi_test["dmd_wkly_dos"].transform(lambda x: x.fillna(x.median()))


dfi["mdi_stockouts"] = dfi["mdi_stockouts"].transform(lambda x: x.fillna(1))
dfi_test["mdi_stockouts"] = dfi_test["mdi_stockouts"].transform(lambda x: x.fillna(1))


dfi["network_avail"] = dfi.groupby(["item_id", "pdc"])["network_avail"].transform(lambda x: x.fillna(x.median()))
dfi["network_avail"] = dfi.groupby(["item_id"])["network_avail"].transform(lambda x: x.fillna(x.median()))
dfi["network_avail"] = dfi["network_avail"].transform(lambda x: x.fillna(x.median()))
dfi_test["network_avail"] = dfi_test.groupby(["item_id", "pdc"])["network_avail"].transform(lambda x: x.fillna(x.median()))
dfi_test["network_avail"] = dfi_test.groupby(["item_id"])["network_avail"].transform(lambda x: x.fillna(x.median()))
dfi_test["network_avail"] = dfi_test["network_avail"].transform(lambda x: x.fillna(x.median()))


dfi["ltm_median"] = dfi.groupby(["vndr_concat", "item_id", "pdc"])["ltm_median"].transform(lambda x: x.fillna(x.median()))
dfi["ltm_median"] = dfi.groupby(["vndr_concat", "item_id"])["ltm_median"].transform(lambda x: x.fillna(x.median()))
dfi["ltm_median"] = dfi.groupby(["vndr_concat"])["ltm_median"].transform(lambda x: x.fillna(x.median()))
dfi["ltm_median"] = dfi.groupby(["item_id"])["ltm_median"].transform(lambda x: x.fillna(x.median()))
dfi["ltm_median"] = dfi["ltm_median"].transform(lambda x: x.fillna(x.median()))
dfi_test["ltm_median"] = dfi_test.groupby(["vndr_concat", "item_id", "pdc"])["ltm_median"].transform(lambda x: x.fillna(x.median()))
dfi_test["ltm_median"] = dfi_test.groupby(["vndr_concat", "item_id"])["ltm_median"].transform(lambda x: x.fillna(x.median()))
dfi_test["ltm_median"] = dfi_test.groupby(["vndr_concat"])["ltm_median"].transform(lambda x: x.fillna(x.median()))
dfi_test["ltm_median"] = dfi_test.groupby(["item_id"])["ltm_median"].transform(lambda x: x.fillna(x.median()))
dfi_test["ltm_median"] = dfi_test["ltm_median"].transform(lambda x: x.fillna(x.median()))


dfi["ltm_75pct"] = dfi.groupby(["vndr_concat", "item_id", "pdc"])["ltm_75pct"].transform(lambda x: x.fillna(x.median()))
dfi["ltm_75pct"] = dfi.groupby(["vndr_concat", "item_id"])["ltm_75pct"].transform(lambda x: x.fillna(x.median()))
dfi["ltm_75pct"] = dfi.groupby(["vndr_concat"])["ltm_75pct"].transform(lambda x: x.fillna(x.median()))
dfi["ltm_75pct"] = dfi.groupby(["item_id"])["ltm_75pct"].transform(lambda x: x.fillna(x.median()))
dfi["ltm_75pct"] = dfi["ltm_75pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["ltm_75pct"] = dfi_test.groupby(["vndr_concat", "item_id", "pdc"])["ltm_75pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["ltm_75pct"] = dfi_test.groupby(["vndr_concat", "item_id"])["ltm_75pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["ltm_75pct"] = dfi_test.groupby(["vndr_concat"])["ltm_75pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["ltm_75pct"] = dfi_test.groupby(["item_id"])["ltm_75pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["ltm_75pct"] = dfi_test["ltm_75pct"].transform(lambda x: x.fillna(x.median()))


dfi["ltm_90pct"] = dfi.groupby(["vndr_concat", "item_id", "pdc"])["ltm_90pct"].transform(lambda x: x.fillna(x.median()))
dfi["ltm_90pct"] = dfi.groupby(["vndr_concat", "item_id"])["ltm_90pct"].transform(lambda x: x.fillna(x.median()))
dfi["ltm_90pct"] = dfi.groupby(["vndr_concat"])["ltm_90pct"].transform(lambda x: x.fillna(x.median()))
dfi["ltm_90pct"] = dfi.groupby(["item_id"])["ltm_90pct"].transform(lambda x: x.fillna(x.median()))
dfi["ltm_90pct"] = dfi["ltm_90pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["ltm_90pct"] = dfi_test.groupby(["vndr_concat", "item_id", "pdc"])["ltm_90pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["ltm_90pct"] = dfi_test.groupby(["vndr_concat", "item_id"])["ltm_90pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["ltm_90pct"] = dfi_test.groupby(["vndr_concat"])["ltm_90pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["ltm_90pct"] = dfi_test.groupby(["item_id"])["ltm_90pct"].transform(lambda x: x.fillna(x.median()))
dfi_test["ltm_90pct"] = dfi_test["ltm_90pct"].transform(lambda x: x.fillna(x.median()))


dfi["ltm_90pct_difference_wks"] = dfi.groupby(["vndr_concat", "item_id", "pdc"])["ltm_90pct_difference_wks"].transform(lambda x: x.fillna(x.median()))
dfi["ltm_90pct_difference_wks"] = dfi.groupby(["vndr_concat", "item_id"])["ltm_90pct_difference_wks"].transform(lambda x: x.fillna(x.median()))
dfi["ltm_90pct_difference_wks"] = dfi.groupby(["vndr_concat"])["ltm_90pct_difference_wks"].transform(lambda x: x.fillna(x.median()))
dfi["ltm_90pct_difference_wks"] = dfi.groupby(["item_id"])["ltm_90pct_difference_wks"].transform(lambda x: x.fillna(x.median()))
dfi["ltm_90pct_difference_wks"] = dfi["ltm_90pct_difference_wks"].transform(lambda x: x.fillna(x.median()))
dfi_test["ltm_90pct_difference_wks"] = dfi_test.groupby(["vndr_concat", "item_id", "pdc"])["ltm_90pct_difference_wks"].transform(lambda x: x.fillna(x.median()))
dfi_test["ltm_90pct_difference_wks"] = dfi_test.groupby(["vndr_concat", "item_id"])["ltm_90pct_difference_wks"].transform(lambda x: x.fillna(x.median()))
dfi_test["ltm_90pct_difference_wks"] = dfi_test.groupby(["vndr_concat"])["ltm_90pct_difference_wks"].transform(lambda x: x.fillna(x.median()))
dfi_test["ltm_90pct_difference_wks"] = dfi_test.groupby(["item_id"])["ltm_90pct_difference_wks"].transform(lambda x: x.fillna(x.median()))
dfi_test["ltm_90pct_difference_wks"] = dfi_test["ltm_90pct_difference_wks"].transform(lambda x: x.fillna(x.median()))


# Read the training data and assigning feature and target variables
X_train = dfi.iloc[:, 4:-1]   # features
y_train = dfi.iloc[:, -1]     # target variable
print(f'\nShape of the original feature data: {X_train.shape}')
featureNames = dfi.columns.values[4:-1]
print(featureNames)
(dfi['rhit_label'] == 0).sum()
(dfi['rhit_label'] == 1).sum()

# Read the testing data and assigning feature and target variables
X_test = dfi_test.iloc[:, 4:-1]   # features
y_test = dfi_test.iloc[:, -1]     # target variable

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) 
X_test = scaler.fit_transform(X_test) 

# Create an instance PCA and build the model using X_train
pca_prep = PCA().fit(X_train)
pca_prep.n_components_

# Now, we need to determine how many transformed features (or primary components) we want to use in our model.
# PCA provides an array of variances (corresponding to amount of info) of each component in descending order. 
# We inspect those numbers and draw a scree plot. Consider the variances as amount of information.  
# We drop those components providing less information (low variances)

# We have 46 components. Using PCA to find out how many components to use without losing much information.
pca_prep.explained_variance_
pca_prep.explained_variance_ratio_

# A scree plot will help us understand the variances using cumulative ratios
plt.plot(np.cumsum(pca_prep.explained_variance_ratio_))
plt.xlabel('k number of components')
plt.ylabel('cumulative explained variance')
plt.grid(True)
plt.show()

# Perform PCA on the training data
pca = PCA(n_components = 20)
X_train_pca = pca.fit_transform(X_train)

# Perform PCA on the test data using the same PCA object
X_test_pca = pca.transform(X_test)

# Create a Random Forest Classifier
rfmc = RandomForestClassifier(n_estimators = 100, min_samples_split = 10, 
                              min_samples_leaf = 2, criterion = 'entropy')

# Use cross_val_predict on the training data to obtain predicted labels
y_pred_train = cross_val_predict(rfmc, X_train_pca, y_train, cv=5)

# # Fit the model on the entire training data
rfmc.fit(X_train_pca, y_train)

# Predict the labels for the test data
y_pred_test = rfmc.predict(X_test_pca)

# Generate the Random Forest classification report for the Holdout data
rfmc_report_test = classification_report(y_test, y_pred_test)
print("Random Forest Classification Report for the Test data:")
print(rfmc_report_test)


