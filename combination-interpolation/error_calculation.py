import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from DST.loader import Loader


def smape2(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100


def error_calc(city, indicator, method, filled_results_file, eTag):
    data_before_masked = pd.read_csv(r'data/' + eTag +'/'+city + '-' + indicator + '-extracted.csv')
    loader = Loader(city, indicator, eTag)
    _, _, _, _, _, _, X_test_row_index, X_test_column_index = loader.train_test_loader(0.3, 128)

    filled_Results = pd.read_csv(filled_results_file)

    dataset = iter([filled_Results, data_before_masked])
    data = []

    for item in dataset:
        data.append([item.iloc[int(i), int(j)] for i, j in zip(X_test_row_index, X_test_column_index)])

    filled_data = np.array(data[0])
    real_data = np.array(data[1])
    smape1 = smape2(real_data, filled_data)
    # print("SMAPE:%.2f", smape1)
    mae = mean_absolute_error(real_data, filled_data)
    # print("mean_absolute_error:", mae)
    mse = np.sum((real_data - filled_data) ** 2) / len(real_data)
    rmse = math.sqrt(mse)

    saveError(city, indicator, method, rmse, mae, smape1, eTag)

    return filled_data, real_data



def saveError(city, indicator, method, rmse, mae, smape, eTag):
    try:
        finalResults = pd.read_csv('results\\' + eTag +'/'+city + '-' + indicator + '-finalResults.csv')
    except:
        finalResults = pd.DataFrame(np.zeros([10, 4]),
                                    columns=['Method', 'MAE', 'RMSE', 'SMAPE'])

    mindex = {'LT': 0, 'OK': 1, 'RST': 2, 'EST': 3, 'EFIM': 4, 'HST': 5, 'STISE': 6, 'BiLSTM': 7, 'CGDST': 8, 'FGDST': 9}


    finalResults.loc[mindex[method], 'Method'] = method
    finalResults.loc[mindex[method], 'RMSE'] = rmse
    finalResults.loc[mindex[method], 'MAE'] = mae
    finalResults.loc[mindex[method], 'SMAPE'] = smape

    finalResults = finalResults[['Method', 'MAE', 'RMSE', 'SMAPE']]
    finalResults.to_csv('results/' + eTag +'/'+city + '-' + indicator + '-finalResults.csv', index=False)
    # print(city, " is done!")