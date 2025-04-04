#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.pipeline import Pipeline

import logging

logger = logging.getLogger(__name__)

data_path = "./data/california_housing.csv"

def read_datafram(data_path):

    data = pd.read_csv(data_path)

    logger.debug(f"DF shape: {data.shape}")

    data = data.drop(columns="total_bedrooms")
    
    return data

def train(data):

    df= read_dataframe(data_path)
    data_train, data_test = train_test_split(df, test_size=0.33, random_state=0)

    X_train = data_train.drop(columns="median_house_value")
    y_train = data_train["median_house_value"]
    X_test = data_test.drop(columns="median_house_value")
    y_test = data_test["median_house_value"]

    pipeline = make_pipeline(StandardScaler(), LinearRegression())
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mse = metrics.mean_squared_error(y_test, y_pred)
    logger.info(f"MSE: {mse}")

    with open(out_path, "wb") as f_out:
        pickle.dump(pipeline, f_out)
    
    return mse

