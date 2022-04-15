from warnings import warn

import numpy as np
import pandas as pd
from pykrige import OrdinaryKriging
from tqdm import tqdm

from DataConstants import DataConstants
from process_data import checkNan


class Kriging_infill:
    def __init__(self, location_file, data_file, city, indicator, stations):
        self.locationFile = location_file
        self.dataFile = data_file
        self.longList, self.latiList = self.get_local()
        self.data = pd.read_csv(self.dataFile)
        self.stations = stations

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

            # Find the null value and return the list of air stations with null value nanlist
            # and the list of air stations without null value unnanlist
            nanList, unNanList = checkNan(r, self.stations)

            if not nanList:  # If the list of air stations with empty values is empty, interpolation is not required
                continue
            if not unNanList:
                warn("Warning: " + str(id) + "has no data")
                continue


            nan_lon_list = []
            nan_lat_list = []
            for nan_point in nanList:
                nan_lon_list.append(self.longList[nan_point])
                nan_lat_list.append(self.latiList[nan_point])

            unnan_lon_list = []
            unnan_lat_list = []
            for unNan_point in unNanList:
                unnan_lon_list.append(self.longList[unNan_point])
                unnan_lat_list.append(self.latiList[unNan_point])

            # obtain data of non-null air stations
            unnan_z_list = r[unNanList]

            try:
                # Create ordinary kriging object:
                OK = OrdinaryKriging(
                    unnan_lon_list,
                    unnan_lat_list,
                    unnan_z_list,
                    variogram_model="linear",
                    verbose=False,
                    enable_plotting=False,
                    coordinates_type="geographic",
                )

                # Execute on grid:
                z1, ss1 = OK.execute("grid", nan_lon_list, nan_lat_list)

                z2 = np.diagonal(np.array(z1))
                ss2 = np.diagonal(np.array(ss1))

                for point, value, sigma2 in zip(nanList, z2, ss2):
                    self.data.loc[i, point] = value
            except ValueError:
                for point in nanList:
                    self.data.loc[i, point] = r[unNanList].mean()


def ok(eTag):
    dc = DataConstants(eTag)
    cities, indicator, stations = dc.getPublicData()

    i = 0
    for city in cities:
        location_file = r"./data/" + city + "-station-location.csv"
        data_file = r'./data/' + eTag + '/' + city + '-' + indicator + '-masked.csv'
        infilled_file = "./results/" + eTag + '/' + city + "-" + indicator + "-ok-filling.csv"

        gsp_tql = Kriging_infill(location_file, data_file, city, indicator, stations[i])
        gsp_tql.infill_krige()
        gsp_tql.data.to_csv(infilled_file, index=False)
        i = i + 1
