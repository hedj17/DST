import datetime
import random
import warnings
from statistics import mean
from typing import Dict, List

import torch
import torch.nn as nn
from scipy.stats import rv_discrete
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math
import os
import pickle

smape = lambda x, y: 2.0 * np.mean(np.abs(x - y) / (np.abs(x) + np.abs(y))) * 100
warnings.filterwarnings("ignore")

config = {
    # Hyperparameters of spatiotemporal interpolation and spatial interpolation
    "max_k": 10,
    "gamma": 0.1,
    "gamma_*": 0.1,
    "gaussian_weight": 1,
    "seed": 7,  # 7

    # data
    "min_length": 10,  # Minimum continuous vacancy free length for training and testing
    "p_fault": 0.04,
    "t_repair_lam": 5,
    "bias": 2,
    "r_distribution": {
        0: 0.25, 1: 0.22, 2: 0.17, 3: 0.12, 4: 0.08, 5: 0.06, 6: 0.04, 7: 0.02, 8: 0.02, 9: 0.02
    },
    "dataset_path": "data/stise_dataset.pkl",

    # elm
    "hidden_dim": 16,

    # training
    "train_mode": "sgd",
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "batch_size": 128,
    "epoch": 2000,
    "lr": 1e-4
}


def get_tkb_tkf(raw_data: pd.DataFrame, k, index, sites_list: List[str]):
    """get tj, tkf, tkb"""
    x = pd.DataFrame()
    assert k != 0
    k_index = index + k
    x = x.append(raw_data[sites_list].iloc[index])
    x = x.append(raw_data[sites_list].iloc[k_index])
    x = x.dropna(axis=1)
    return x.iloc[0].values, x.iloc[1].values


def window_obj_func(raw_data: pd.DataFrame, k: int, index, sites_list: List[str]):
    """get object function to calculate the size of window
       Where k > 0 corresponds to the future and K < 0 corresponds to history"""
    res = []
    for i in range(1, abs(k) + 1):
        i = i if k > 0 else -i
        if i + index > max(raw_data.index) or i + index < min(raw_data.index):
            break
        tj, tk = get_tkb_tkf(raw_data=raw_data, k=i, index=index, sites_list=sites_list)
        res.append(np.cov(tj, tk)[1][0] / (np.sqrt(np.std(tj)) * np.sqrt(np.std(tk))))
    return res


def select_time_window(raw_data: pd.DataFrame, index, sites_list: List[str]):
    """Get the best window value in a certain data and in a specific site"""
    # take the best history window size
    if isinstance(index, datetime.datetime):
        k = datetime.timedelta(hours=config['max_k'])
    else:
        k = config['max_k']

    max_kf_func = - float("inf")
    kf = 0
    kf_func = window_obj_func(raw_data=raw_data, k=k, index=index, sites_list=sites_list)
    for i in range(1, len(kf_func) + 1):
        kf_temp = mean(kf_func[: i])
        if kf_temp > max_kf_func:
            kf = i
            max_kf_func = kf_temp
    del max_kf_func

    # get the best future window size
    if isinstance(index, datetime.datetime):
        k = datetime.timedelta(hours=-config['max_k'])
    else:
        k = - config['max_k']

    max_kb_func = - float("inf")
    kb = 0
    kb_func = window_obj_func(raw_data=raw_data, k=k, index=index, sites_list=sites_list)
    for i in range(1, len(kb_func) + 1):
        kb_temp = mean(kb_func[: i])
        if kb_temp > max_kb_func:
            kb = i
            max_kb_func = kb_temp
    del max_kb_func

    return kb, kf


def get_weight(k: int):
    return config["gamma_*"] * math.pow(1 - config["gamma"], k - 1)


def improved_ses_all(raw_data: pd.DataFrame, site_list: List[str]):
    data = raw_data.copy(deep=True)
    kb, kf = {}, {}
    for i in data.index:
        if data.iloc[i].isnull().any():
            b, f = select_time_window(raw_data, i, site_list)
            kb[i] = b
            kf[i] = f
    data = raw_data.copy(deep=True)
    for site in site_list:
        seq = data[site].copy(deep=True)
        for i in range(len(seq)):
            i = i + data.index[0]
            if not np.isnan(seq[i]):
                continue

            weight_dict = {}

            # gets the historical value and the weight of the historical value
            for k in range(1, kb[i] + 1):
                try:
                    value = seq[i - k]
                except KeyError:
                    continue
                if np.isnan(value):
                    continue
                weight_dict[value] = weight_dict.get(value, 0) + get_weight(k)

            # gets future values and weights for future values
            for k in range(1, kf[i] + 1):
                try:
                    value = seq[i + k]
                except KeyError:
                    continue
                if np.isnan(value):
                    continue
                weight_dict[value] = weight_dict.get(value, 0) + get_weight(k)

            if len(weight_dict) > 0:
                estimated_value = 0.0
                for k, v in weight_dict.items():
                    estimated_value += k * v
                estimated_value /= sum(weight_dict.values())
                seq[i] = estimated_value
            else:
                warnings.warn(f"Position {i} interpolation failed because the window is empty")
        data[site] = seq.values
    return data


def cal_r_i_k(raw_data: pd.DataFrame, i: str, k: str):
    seq = raw_data[[i, k]]
    seq = seq.dropna().values
    si, sk = seq[:, 0], seq[:, 1]
    return np.cov(si, sk)[0][1] / (np.std(si) * np.std(sk))


def get_distance(sites_location: pd.DataFrame, point_list: List[str] = None):
    if point_list is None:
        point_list = [p for p in sites_location if 'A' in sites_location]
    sites_location = sites_location.set_index("监测点编码")
    distance = {p: {} for p in sites_location.index}
    for pi in sites_location.index:
        if pi not in point_list:
            continue
        for pk in sites_location.index:
            if pk not in point_list:
                continue
            if pi == pk:
                distance[pi][pk] = 0
            elif pi not in distance.keys() or pk not in distance[pi].keys():
                lng1 = sites_location["经度"][pi] * math.pi / 180
                lng2 = sites_location["经度"][pk] * math.pi / 180
                lat1 = sites_location["纬度"][pi] * math.pi / 180
                lat2 = sites_location["纬度"][pk] * math.pi / 180
                cos = math.cos(lat2) * math.cos(lat1) * math.cos(lng2 - lng1) + math.sin(lat1) * math.sin(lat2)
                d = math.acos(cos)
                distance[pi][pk] = 6357 * d
                distance[pk][pi] = 6357 * d
    return distance


def cal_dist_i_k(raw_data: pd.DataFrame, distance: Dict[str, Dict[str, float]], i: str, k: str):
    four_pia2 = 4 * math.pi * math.pow(config["gaussian_weight"], 2)
    v = - math.pow(distance[i][k], 1 - float(cal_r_i_k(raw_data, i, k)))
    return 1 / four_pia2 * math.exp(v / four_pia2)


def check_nan(row: pd.Series, point_list):
    """
    :param row: Data of multiple sites at a certain time
    :param point_list: list of the sites' name
    :return (nan_list: The site where the time data is vacant
                un_nan_list: The site where the time data is vacant )
    """
    nan_list = []
    un_nan_list = []
    for i in row.axes[0]:
        if i not in point_list:
            continue
        if pd.isnull(row[i]):
            nan_list.append(i)
        else:
            un_nan_list.append(i)
    return nan_list, un_nan_list


def improved_idw(raw_data: pd.DataFrame, sites_location: pd.DataFrame, point_list: List[str]):
    # calculate the distance between each site and the weight between each site
    distance = get_distance(sites_location=sites_location, point_list=point_list)
    dist_i_k_dict = {p: {} for p in point_list}
    for pi in point_list:
        for pk in point_list:
            if pi == pk:
                dist_i_k_dict[pi][pk] = 0
            elif pi not in dist_i_k_dict.keys() or pk not in dist_i_k_dict[pi].keys():
                dist_i_k = cal_dist_i_k(raw_data=raw_data, distance=distance, i=pi, k=pk)
                dist_i_k_dict[pi][pk] = dist_i_k
                dist_i_k_dict[pk][pi] = dist_i_k

    infilled_data = pd.DataFrame(columns=raw_data.columns)
    for i, r in raw_data.iterrows():
        # # add datetime
        # t = datetime.datetime.strptime(str(int(r['date'])) + str(int(r['hour'])), '%Y%m%d%H')
        # r['datetime'] = t

        # Find null values and return the list of sites where the null value is located
        nan_list, un_nan_list = check_nan(r, point_list)
        if not nan_list:
            infilled_data = infilled_data.append(r)
            continue

        # If all site data is empty, will be unable to work
        if not un_nan_list:
            infilled_data = infilled_data.append(r)
            warnings.warn("Warning: " + str(i) + "has no data")
            continue

        # interpolate the vacancy value
        for nan_point in nan_list:
            weight_dict = {}
            res = 0.0
            for un_nan_point in un_nan_list:
                weight_dict[r[un_nan_point]] = weight_dict.get(r[un_nan_point], 0) + dist_i_k_dict[nan_point][
                    un_nan_point]
                res += r[un_nan_point] * dist_i_k_dict[nan_point][un_nan_point]
            r[nan_point] = res / sum(weight_dict.values())
        infilled_data = infilled_data.append(r)

    return infilled_data


def get_continuous_data(raw_df: pd.DataFrame, min_length: int, effective_columns: List[str]) -> List[pd.DataFrame]:
    """
    :func A certain length of row without missing values is reserved for training the model
    :param raw_df: Data set to be interpolated
    :param min_length: Minimum continuous length of data set
    :param effective_columns: Valid columns available for interpolation
    :return Data available for training
    """
    # Get row with null value
    df_isnull = raw_df[effective_columns].isnull().any(axis=1)
    null_row_list = [i for i in range(len(df_isnull)) if df_isnull[i]] + [len(raw_df)]

    # Get no vacancy for longer than min_ Length data
    train_test_dataset = []
    last_null_row_index = -1
    for null_row_index in null_row_list:
        if null_row_index - last_null_row_index > min_length:
            train_test_dataset.append(
                raw_df[effective_columns].iloc[last_null_row_index + 1: null_row_index - 1])
        last_null_row_index = null_row_index
    return train_test_dataset


def mask_point(raw_df: pd.DataFrame, effective_columns: List[str]):
    """
    :func Masking continuous data for interpolation
    :param effective_columns: The name of a valid columns that can be used to empty
    :param raw_df: Original data set to be interpolated
    :return: Random mask matrix, where the mask is changed to NP nan
    """
    raw_df = raw_df.copy(deep=True)

    mask_matrix = np.random.random(size=raw_df[effective_columns].shape)
    mask_matrix[config['bias']: -config['bias']][
        np.where(mask_matrix[config['bias']: -config['bias']] < config["p_fault"])] = 1

    r_distribution = rv_discrete(
        values=(np.array(list(config['r_distribution'].keys())), np.array(list(config['r_distribution'].values()))))
    number_of_missing_values = r_distribution.rvs(size=np.count_nonzero(mask_matrix == 1))

    for i, (fault_x, fault_y) in enumerate(zip(*np.where(mask_matrix == 1))):
        if fault_x + number_of_missing_values[i] < len(raw_df) - config["bias"]:
            mask_matrix[fault_x: fault_x + number_of_missing_values[i], fault_y] = 1
    mask_matrix[np.where(mask_matrix != 1)] = 0

    # Assign the position assigned with 1 to Nan
    masked_df_values = raw_df[effective_columns].values
    masked_df_values = masked_df_values.astype(float)
    masked_df_values[np.where(mask_matrix == 1)] = np.nan
    raw_df[effective_columns] = masked_df_values
    masked_df = raw_df.copy(deep=True)
    return masked_df


def get_row_null(raw_df: pd.DataFrame):
    """
    :func Obtain rs in data
    :param raw_df: Data to be interpolated
    :return: rs in data
    """
    row_nul = pd.DataFrame(columns=raw_df.columns)
    col = raw_df.columns
    for i, r in raw_df.iterrows():
        row_nul_row = {}
        for c in col:
            row_nul_row[c] = r.isnull().sum()
        row_nul = row_nul.append(row_nul_row, ignore_index=True)
    return row_nul


def get_train_test_set(continuous_data_list: List[pd.DataFrame], sites_location: pd.DataFrame, point_list: List[str]):
    """Get the total data of training ElmNet"""
    temp_result = []
    spat_result = []
    null = []
    y = []
    for continuous_data in continuous_data_list:
        mask_df = mask_point(continuous_data, effective_columns=point_list)
        null.append(mask_df)
        temp_result.append(improved_ses_all(mask_df, site_list=point_list))
        spat_result.append(improved_idw(raw_data=mask_df, sites_location=sites_location, point_list=point_list))
        y.append(continuous_data)

    null = pd.concat(null)
    temp_result = pd.concat(temp_result)
    spat_result = pd.concat(spat_result)
    y = pd.concat(y)

    dataset = iter([temp_result, spat_result])
    data = []
    for item in dataset:
        data.append([item.iloc[i, j] for i, j in zip(np.where(pd.isnull(null))[0], np.where(pd.isnull(null))[1])])
    data = np.array(data).T
    target = np.array([y.iloc[i, j] for i, j in zip(np.where(pd.isnull(null))[0], np.where(pd.isnull(null))[1])])
    return np.concatenate([data, np.expand_dims(target, axis=1)], axis=1)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ElmNet(nn.Module):
    def __init__(self):
        super(ElmNet, self).__init__()
        self.inp2hid = nn.Linear(2, config["hidden_dim"], bias=True)
        self.act = nn.Sigmoid()
        self.hid2out = nn.Linear(config["hidden_dim"], 1, bias=False)
        self.weight_init()

    def weight_init(self):
        setup_seed(config["seed"])
        torch.nn.init.torch.nn.init.uniform_(self.inp2hid.bias, -0.4, 0.4)
        torch.nn.init.xavier_uniform_(self.inp2hid.weight, gain=1)
        torch.nn.init.xavier_uniform_(self.hid2out.weight, gain=1)

    def fine_ture(self, x, y):
        # optimization using extreme learning machine
        m, s = x.mean(), x.std()
        x = (x - m) / s
        y = (y - m) / s
        y = y.unsqueeze(dim=0).to(torch.float) if y.dim() < 2 else y.to(torch.float)
        hid_out = self.act(self.inp2hid(x.to(torch.float)))
        self.hid2out.state_dict()['weight'] = torch.linalg.lstsq(hid_out, y.permute(1, 0)).solution

    def forward(self, x):
        x = x.to(torch.float)
        m, s = x.mean(), x.std()
        x = (x - m) / s
        y = self.hid2out(self.act(self.inp2hid(x)))
        return torch.tensor(y * s + m)


class ElmDataSet(Dataset):
    def __init__(self, dataset):
        super(ElmDataSet, self).__init__()
        self.x = dataset[:, :2]
        self.y = dataset[:, 2]

    def __getitem__(self, item):
        return torch.from_numpy(self.x[item]), torch.tensor(self.y[item])

    def __len__(self):
        return len(self.y)


def get_elm_model(dataset: np.ndarray):
    # get data
    x, y = dataset[:, :2], dataset[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    elm_net = ElmNet()
    elm_net.fine_ture(torch.from_numpy(x_train), torch.from_numpy(y_train))

    # test model
    elm_net.eval()
    with torch.no_grad():
        rmse = mean_squared_error(y_test, elm_net.forward(torch.from_numpy(x_test)).numpy(), squared=False)

    return elm_net, rmse


def st_ise(raw_data: pd.DataFrame, sites_location: pd.DataFrame, site_list: List[str]):
    if not os.path.exists(config["dataset_path"]):
        continuous_data_list = get_continuous_data(raw_data, min_length=50, effective_columns=site_list)
        dataset = get_train_test_set(continuous_data_list, sites_location, site_list)
        pickle.dump(dataset, open(config["dataset_path"], 'wb'))
    else:
        dataset = pickle.load(open(config["dataset_path"], 'rb'))
    elm, _ = get_elm_model(dataset)

    # The temporal interpolation results and spatial interpolation results are obtained respectively
    temp_result = improved_ses_all(raw_data, site_list=site_list)
    spat_result = improved_idw(raw_data=raw_data, sites_location=sites_location, point_list=site_list)

    # interpolate
    infilled_data = pd.DataFrame(columns=site_list + ['datetime'])
    raw_data = raw_data[site_list]
    for i, r in raw_data.iterrows():
        # Find null values and return the list of sites where the null value is located
        nan_list, un_nan_list = check_nan(r, site_list)
        if not nan_list:
            infilled_data = infilled_data.append(r)
            continue

        # If all site data is empty, only use temporal infill
        if not un_nan_list:
            r[nan_list] = temp_result[nan_list].iloc[i]
            infilled_data = infilled_data.append(r)
            continue

        # Interpolate the missing value
        for nan_point in nan_list:
            x = torch.tensor([temp_result[nan_point][i], spat_result[nan_point][i]])
            r[nan_point] = float(elm.forward(x))
        infilled_data = infilled_data.append(r)

    return infilled_data

#
# if __name__ == '__main__':
#     raw_df = pd.read_csv(r"data/xian-PM10-raw.csv")
#     sites_location = pd.read_csv(r"data/xian-station-location.csv", encoding='gbk')
#     site_list = ['1462A', '1463A', '1464A', '1465A', '1466A', '1467A', '1468A', '1469A', '1471A', '1472A', '1473A',
#                  '1474A']
#     infilled = st_ise(raw_data=raw_df, site_list=site_list, sites_location=sites_location)
#     print(infilled)
#     infilled.to_csv("data/infilled.csv")
