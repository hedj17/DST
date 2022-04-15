import os
import warnings
from typing import Tuple, Dict, List

from torch.utils.data import DataLoader
import torch

from BiLSTM.loader import Scaler, DataSet
from BiLSTM.model import LSTM
from BiLSTM.loader import get_distance
from DST.loader import Loader
from DataConstants import DataConstants
from BiLSTM.config import config
import numpy as np
from BiLSTM.train import train as get_model
import pandas as pd
import pickle


def get_all_masked(masked_data: pd.DataFrame) -> List[Tuple[int, int]]:
    shp = masked_data.shape
    row_len, col_len = shp[0], shp[1]
    result = []
    for i in range(row_len):
        for j in range(col_len):
            x = masked_data.iloc[i, j]
            if not isinstance(x, float):
                continue
            if (np.isnan(masked_data.iloc[i, j])):
                result.append((i, j))
    return result


def index2sample(x_index: Tuple[int, int], masked_data, extracted_data, distance: Dict[str, Dict[str, float]]):
    """Obtain trainable or predictable samples according to the row and column index"""
    row_index, col_index = x_index
    site_name = masked_data.columns[col_index]
    site_list = [p for p in masked_data.columns if 'A' in p]

    # find all the sites that are not empty at the first and last k times at the specified time
    t_h = row_index - config.t
    t_f = row_index + config.t + 1
    data = masked_data[site_list].iloc[t_h: t_f, :]
    if len(data) < 2 * config.t + 1:
        return None

    site_candidate = []
    for site in site_list:
        if np.any(data[site].isnull()):
            continue
        site_candidate.append(site)

    # select k nearest sites
    if len(site_candidate) < config.k:
        warnings.warn(f"Insufficient available adjacent sites: Row coordinates{row_index} Column coordinates{col_index}")
        return None
    distance = {k: distance[site_name][k] for k in site_candidate}
    site_candidate = [s[0] for s in sorted(distance.items(), key=lambda e: e[1])[: config.k]]
    return data[site_candidate].values, float(extracted_data.iloc[row_index, col_index])


def bilstm(eTag):
    dc = DataConstants(eTag)
    cities, indicator, stations = dc.getPublicData()

    for city in cities:
        loader = Loader(city, indicator, eTag)
        _, __, ___, ____, _____, ______, X_test_row_index, X_test_column_index = loader.train_test_loader(
            0.3, 128)

        masked_data = pd.read_csv(r'./data/' + eTag + '/' + city + '-' + indicator + '-masked.csv')
        extracted_data = pd.read_csv(r'./data/' + eTag + '/' + city + '-' + indicator + '-extracted.csv')
        if eTag in ['e84', 'e86']:
            location_data = pd.read_csv(r'./data/' + city + "-station-location-selected.csv")
        else:
            location_data = pd.read_csv(r'./data/' + city + "-station-location.csv")

        """get train dataset by X_test_row_index, X_test_column_index in masked data"""
        # get index of train dataset and test dataset
        X_test_row_index, X_test_column_index = \
            X_test_row_index.astype(int).tolist(), X_test_column_index.astype(int).tolist()
        X_test_index = [(r, c) for r, c in zip(X_test_row_index, X_test_column_index)]
        X_train_index = list(set(get_all_masked(masked_data)) - set(X_test_index))

        # get train dataset and test dataset by index
        distance = get_distance(location_data, [p for p in masked_data.columns if 'A' in p])

        if os.path.exists(config.train_loader_path) and os.path.exists(config.test_loader_path):
            train_loader = pickle.load(open(config.train_loader_path, "rb"))
            test_loader = pickle.load(open(config.test_loader_path, "rb"))
            scaler = pickle.load(open(config.scaler_path, "rb"))

        else:
            train_x_set = []
            train_y_set = []
            for train_index in X_train_index:
                xy = index2sample(
                    x_index=train_index,
                    masked_data=masked_data,
                    extracted_data=extracted_data,
                    distance=distance
                )
                if xy is None:
                    continue
                train_x_set.append(xy[0])
                train_y_set.append(xy[1])

            test_x_set = []
            test_y_set = []
            for test_index in X_test_index:
                xy = index2sample(
                    x_index=test_index,
                    masked_data=masked_data,
                    extracted_data=extracted_data,
                    distance=distance
                )
                if xy is None:
                    continue
                test_x_set.append(xy[0])
                test_y_set.append(xy[1])

            train_x_set = np.array(train_x_set)
            train_y_set = np.array(train_y_set)
            test_x_set = np.array(test_x_set)
            test_y_set = np.array(test_y_set)

            # standardization
            scaler = Scaler(np.concatenate([train_x_set.flatten(), train_y_set.flatten()]))
            train_x_set = scaler.transform(train_x_set)
            train_y_set = scaler.transform(train_y_set)
            test_x_set = scaler.transform(test_x_set)
            test_y_set = scaler.transform(test_y_set)

            # build dataset
            train_set = DataSet(train_x_set, train_y_set)
            test_set = DataSet(test_x_set, test_y_set)
            train_loader = DataLoader(train_set, batch_size=config.batch_size)
            test_loader = DataLoader(test_set, batch_size=config.batch_size)
            pickle.dump(train_loader, open(config.train_loader_path, "wb"))
            pickle.dump(test_loader, open(config.test_loader_path, "wb"))
            pickle.dump(scaler, open(config.scaler_path, "wb"))

        # train model
        if config.model_path is not None and os.path.exists(config.model_path):
            model = torch.load(config.model_path)
        else:
            model = LSTM(config).to(config.device)
            model.load_state_dict(get_model(config, train_loader, test_loader, scaler))
        model.eval()

        # Interpolate the test set through the model,
        # and then write the results to masked_data or another dataframe,
        # its size should be the same as masked_data is exactly the same
        with torch.no_grad():
            for test_index in X_test_index:
                xy = index2sample(
                    x_index=test_index,
                    masked_data=masked_data,
                    extracted_data=extracted_data,
                    distance=distance
                )
                if xy is None:
                    continue
                else:
                    masked_data.iloc[test_index[0], test_index[1]] = scaler.inverse_transform(
                        float(model(torch.from_numpy(scaler.transform(xy[0])).to(torch.float32))))

        # Using average interpolation
        site_list = [p for p in masked_data.columns if 'A' in p]
        for i, r in masked_data.iterrows():
            # Gets null and non-null values in a row
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

        for train_index in X_train_index:
            row_index, col_index = train_index
            masked_data.iloc[row_index, col_index] = np.nan

        infilled_file = "./results/" + eTag + '/' + city + "-" + indicator + "-bilstm-filling.csv"

        masked_data.to_csv(infilled_file, index=False)
