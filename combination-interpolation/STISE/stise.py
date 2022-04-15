from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch

from BiLSTM.bilstm import get_all_masked
from DST.loader import Loader
from DataConstants import DataConstants
from STISE.module import improved_ses_all, improved_idw, ElmNet


def stise(eTag):
    dc = DataConstants(eTag)
    cities, indicator, stations = dc.getPublicData()

    for city in cities:
        loader = Loader(city, indicator, eTag)
        _, __, ___, ____, _____, ______, X_test_row_index, X_test_column_index = loader.train_test_loader(
            0.3, 128)

        masked_data = pd.read_csv(r'./data/' + eTag + '/' + city + '-' + indicator + '-masked.csv')
        extracted_data = pd.read_csv(r'./data/' + eTag + '/' + city + '-' + indicator + '-extracted.csv')
        # the furthest station is removed in the experiment 'e84'
        if eTag in ['e84', 'e86']:
            location_data = pd.read_csv(r'./data/' + city + "-station-location-selected.csv")
        else:
            location_data = pd.read_csv(r'./data/' + city + "-station-location.csv")

        # get indexes of training data and test data respectively
        X_test_row_index, X_test_column_index = \
            X_test_row_index.astype(int).tolist(), X_test_column_index.astype(int).tolist()
        X_test_index = [(r, c) for r, c in zip(X_test_row_index, X_test_column_index)]
        X_train_index = list(set(get_all_masked(masked_data)) - set(X_test_index))

        # Transform indexes into training sets and test sets
        """Obtain trainable or predictable samples according to the row and column index"""
        site_list = [p for p in masked_data.columns if 'A' in p]
        temp_result = improved_ses_all(masked_data, site_list=site_list)
        spat_result = improved_idw(raw_data=masked_data, sites_location=location_data, point_list=site_list)
        x_train, y_train = [], []
        for x_idx in X_train_index:
            row_index, col_index = x_idx
            if np.isnan(temp_result.iloc[row_index, col_index]) or np.isnan(spat_result.iloc[row_index, col_index]):
                continue
            x_train.append([temp_result.iloc[row_index, col_index], spat_result.iloc[row_index, col_index]])
            y_train.append(extracted_data.iloc[row_index, col_index])
        x_train, y_train = torch.tensor(x_train).to(torch.float), torch.tensor(y_train).to(torch.float)

        elm_net = ElmNet()
        elm_net.fine_ture(x_train, y_train)
        elm_net.eval()

        with torch.no_grad():
            for x_idx in X_test_index:
                row_index, col_index = x_idx
                x = torch.tensor([temp_result.iloc[row_index, col_index], spat_result.iloc[row_index, col_index]])
                y = float(elm_net.forward(x))
                masked_data.iloc[row_index, col_index] = y

        site_list = [p for p in masked_data.columns if 'A' in p]
        for i, r in masked_data.iterrows():
            nan_list, un_nan_list = [], []
            for site in site_list:
                if np.isnan(r[site]):
                    nan_list.append(site)
                else:
                    un_nan_list.append(site)
            if len(nan_list) == 0 or len(un_nan_list) == 0:
                continue
            new_r = r.copy(deep=True)
            new_r[nan_list] = float(new_r[un_nan_list].values.mean())
            masked_data.iloc[i, :] = new_r
        # masked_data = masked_data.interpolate(method='linear', limit_direction='both')


        for train_index in X_train_index:
            row_index, col_index = train_index
            masked_data.iloc[row_index, col_index] = np.nan

        infilled_file = "./results/" + eTag + '/' + city + "-" + indicator + "-stise-filling.csv"
        masked_data.to_csv(infilled_file, index=False)
