import math

import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error

from DST.loader import Loader
from DataConstants import DataConstants
from error_calculation import smape2, saveError
import pandas as pd


def dataPreparation(city, indicator, eTag):
    loader = Loader(city, indicator, eTag)
    train_loader, test_loader, X_train, X_test, y_train, y_test, X_test_row_index, X_test_column_index = loader.train_test_loader(
        0.3, 128)
    return X_train, X_test, y_train, y_test, X_test_row_index, X_test_column_index


# linear model
def test_LinearRegrssion(*data):
    x_train, x_test, y_train, y_test = data
    regr = linear_model.LinearRegression()
    # print("The number of nan in x_train is: ", np.isnan(x_train).sum())
    # print("The number of nan in y_train is: ", np.isnan(y_train).sum())
    regr.fit(x_train, y_train)  # train model

    y_predict = regr.predict(x_test)

    smape1 = smape2(y_test, y_predict)
    # print("SMAPE:%.2f", smape1)
    mae = mean_absolute_error(y_test, regr.predict(x_test))
    # print("mean_absolute_error:", mae)
    mse = np.sum((y_test - regr.predict(x_test)) ** 2) / len(y_test)
    rmse = math.sqrt(mse)
    # print("rmse:%.2f" % rmse)
    return rmse, mae, smape1, y_predict


def hst(eTag):
    dc = DataConstants(eTag)
    cities, indicator, stations = dc.getPublicData()

    i = 0

    for city in cities:
        X_train, X_test, y_train, y_test, X_test_row_index, X_test_column_index = dataPreparation(city, indicator, eTag)
        X_train1 = X_train[:, 2:4]
        # Select linear interpolation and OK interpolation results from the input features, because here X_train
        # features have up spacing, down spacing, line spacing, OK interpolation results and linear interpolation
        # results in turn
        X_test1 = X_test[:, 2:4]
        rmse, mae, smape, estimatedValues = test_LinearRegrssion(X_train1, X_test1, y_train, y_test)

        masked_data = pd.read_csv(r'./data/' + eTag + '/' + city + '-' + indicator + '-masked.csv')

        for index in range(len(estimatedValues)):
            masked_data.iloc[int(X_test_row_index[index]), int(X_test_column_index[index])] = estimatedValues[index]

        hst_results_file = r'./results/' + eTag + '/' + city + '-' + indicator + '-hst-filling.csv'
        masked_data.to_csv(hst_results_file, index=False)
