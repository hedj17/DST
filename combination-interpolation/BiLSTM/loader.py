from abc import ABC
from typing import List
import numpy as np
import pandas as pd
import math
import torch
from torch.utils.data import Dataset


def get_distance(sites_location: pd.DataFrame, point_list: List[str] = None):
    """get distance of two sites"""
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


class DataSet(Dataset, ABC):
    def __init__(self, x, y):
        super(DataSet, self).__init__()
        self.x = torch.from_numpy(x).to(torch.float32)
        self.y = torch.from_numpy(y).to(torch.float32)

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.y)


class Scaler:
    def __init__(self, fit_data: np.ndarray):
        self.std = float(fit_data.std())
        self.mean = float(fit_data.mean())

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean
