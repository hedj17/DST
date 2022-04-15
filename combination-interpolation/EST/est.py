from datetime import datetime, timedelta
from warnings import warn

import numpy as np
import pandas as pd
from pykrige.ok3d import OrdinaryKriging3D
from tqdm import tqdm

from DataConstants import DataConstants
from process_data import datetime2str, str2datatime, str2index, checkNan

class TimeFunc:
    def __init__(self, k):
        self.k = k

    def __call__(self, t):
        return t * self.k


class Kriging_infill:
    def __init__(self, location_file, data_file, stations, time_fuc, time_step=1, k_num=2):
        self.locationFile = location_file
        self.dataFile = data_file
        self.stations = stations
        self.longList, self.latiList = self.get_local()
        data = pd.read_csv(self.dataFile)
        data['date'] = data['date'].apply(lambda x: str(x).strip())
        data['hour'] = data['hour'].apply(lambda x: str(x).strip())
        self.data = data

        self.time_step_hour = time_step  # Monitor the data once every hour
        self.k_num = k_num  # Spatiotemporal Kriging interpolation is based on two hours before and after
        self.time_fuc = time_fuc  # Weight of time

    def get_k_neighbor_times(self, now_data_time):
        """Get the k*time_step front of the current time and k*time_step behind of the current time
        :return datetime"""
        k_neighbor_times_list = []
        for i in range(1, self.k_num + 1):
            k_neighbor_times_list.append(now_data_time - timedelta(hours=self.time_step_hour*i))
            k_neighbor_times_list.append(now_data_time + timedelta(hours=self.time_step_hour*i))
        return k_neighbor_times_list + [now_data_time]

    def get_k_neighbour_time_dict(self, now_data_time):
        """Find out the set of all data not missing in the adjacent time period"""
        k_neighbour_time_dict = {}
        for k_neighbour_time in self.get_k_neighbor_times(now_data_time):
            ymd, h = str2index(datetime2str(k_neighbour_time))
            time_row = self.data[(self.data['date'] == ymd) & (self.data['hour'] == h)]
            if time_row.empty:
                continue
            try:
                ts_Series = pd.Series(time_row.values[0], time_row.columns.values)
                _, UnNanPoints = checkNan(ts_Series, self.stations)
            except TypeError:
                continue
            for point in UnNanPoints:
                k_neighbour_time_dict[datetime2str(k_neighbour_time) + '_' + str(point)] = time_row[str(point)]

        return k_neighbour_time_dict

    def get_local(self):
        # return two dict containing the longitude and latitude of the point in the file
        point_loc = pd.read_csv(self.locationFile)
        long = {}
        lati = {}
        for index, row in point_loc.iterrows():
            long[row["监测点编码"]] = row['经度']
            lati[row["监测点编码"]] = row['纬度']
        return long, lati

    def infill_krige(self):

        for i, r in tqdm(self.data.iterrows()):
            # current datetime
            now_data_time = datetime.strptime(str(int(self.data[i: i + 1]['date'].values)) +
                                              str(int(self.data[i: i + 1]['hour'].values)),
                                              '%Y%m%d%H')

            # search null
            nanList, unNanList = checkNan(r, self.stations)
            if not nanList:
                continue

            # Find the number of air stations without missing value in the last K times
            k_neighbour_time_dict = self.get_k_neighbour_time_dict(now_data_time)

            if not k_neighbour_time_dict:
                warn("Warning: " + str(i) + "has no data")
                continue

            # Gets the geographical location of each null air station
            nan_lon_list = []
            nan_lat_list = []
            for nan_point in nanList:
                nan_lon_list.append(self.longList[nan_point])
                nan_lat_list.append(self.latiList[nan_point])

            # Obtain the space-time position of each non-null air station
            unnan_lon_list = []
            unnan_lat_list = []
            unnan_time_list = []
            unnan_z_list = []
            for k, v in k_neighbour_time_dict.items():
                t, p = k.split('_')
                unnan_lon_list.append(self.longList[p])
                unnan_lat_list.append(self.latiList[p])
                t = (str2datatime(t) - now_data_time) / timedelta(hours=self.time_step_hour)
                unnan_time_list.append(self.time_fuc(t))
                unnan_z_list.append(v.values[0])

            try:
                ok3d = OrdinaryKriging3D(
                    unnan_lon_list, unnan_lat_list, unnan_time_list, unnan_z_list, variogram_model="linear"
                )
                k3d1, ss3d = ok3d.execute("grid", nan_lon_list, nan_lat_list, [0])
                k3d2 = np.diagonal(np.array(k3d1))

                for point, value in zip(nanList, k3d2):
                    self.data.loc[i, point] = value
            except ValueError:
                for point in nanList:
                    self.data.loc[i, point] = r[unNanList].mean()

def est(eTag):
    dc = DataConstants(eTag)
    cities, indicator, stations = dc.getPublicData()

    i = 0
    for city in cities:
        location_file = r"./data/" + '/' + city + "-station-location.csv"
        data_file = r'./data/' + eTag + '/' + city + '-' + indicator + '-masked.csv'
        infilled_file = "./results/" + eTag + '/' + city + "-" + indicator + "-est-filling.csv"

        gsp_tql = Kriging_infill(location_file, data_file, stations[i], time_fuc=TimeFunc(10))
        gsp_tql.infill_krige()
        gsp_tql.data.to_csv(infilled_file, index=False)
        i = i + 1
