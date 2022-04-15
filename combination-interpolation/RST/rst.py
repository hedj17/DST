import pandas as pd
from pykrige import OrdinaryKriging
from tqdm import tqdm

from DataConstants import DataConstants
from process_data import checkNan


class krigeAfterLinear_infill:
    def __init__(self, location_file, data_file, stations, lt_results, data_masked, data_before_masked):
        self.locationFile = location_file
        self.dataFile = data_file
        self.longList, self.latiList = self.get_local()
        self.data = pd.read_csv(self.dataFile)
        self.stations = stations
        self.lt_results = lt_results
        self.data_masked = data_masked
        self.data_before_masked = data_before_masked


    def get_local(self):
        # return two dict containing the longitude and latitude of the point in the file
        point_loc = pd.read_csv(self.locationFile)
        long = {}
        lati = {}
        for index, row in point_loc.iterrows():
            long[row["监测点编码"]] = row['经度']
            lati[row["监测点编码"]] = row['纬度']
        return long, lati

    def infill_krigeAfterLinear(self):

        predict_value_list = []
        for i, r in tqdm(self.data.iterrows()):
            nanList, unNanList_null = checkNan(r, self.stations)
            nanList_null, allStationList = checkNan(self.lt_results.iloc[i], self.stations)

            nan_lon_list = []
            nan_lat_list = []
            for nan_point in nanList:
                nan_lon_list.append(self.longList[nan_point])
                nan_lat_list.append(self.latiList[nan_point])

            # Gets the geographical location of each non-null air station
            unnan_lon_list = []
            unnan_lat_list = []

            for unNan_point in allStationList:
                unnan_lon_list.append(self.longList[unNan_point])
                unnan_lat_list.append(self.latiList[unNan_point])

            # obtain the linear interpolation results corresponding to each non-null position
            unnan_z_list = list(self.lt_results.iloc[i][allStationList])

            # The following loop is used to find the forecast data for all the empty values
            count = 0
            for unNan_point in allStationList:
                if unNan_point in nanList:
                    """
                    If the point is empty in the original data, ordinary Kriging 
                    interpolation is performed on the point using the data after linear interpolation (principle of
                    reduction). 
                    """

                    count += 1
                    unnan_lon_list_tmp = unnan_lon_list.copy()
                    unnan_lon_list_tmp.remove(unnan_lon_list[count - 1])
                    unnan_lat_list_tmp = unnan_lat_list.copy()
                    unnan_lat_list_tmp.remove(unnan_lat_list[count - 1])
                    unnan_z_list_tmp = unnan_z_list.copy()
                    unnan_z_list_tmp.remove(unnan_z_list[count - 1])

                    try:
                        # Create ordinary kriging object:
                        OK = OrdinaryKriging(
                            unnan_lon_list_tmp,
                            unnan_lat_list_tmp,
                            unnan_z_list_tmp,
                            variogram_model="linear",
                            verbose=False,
                            enable_plotting=False,
                            coordinates_type="geographic",
                        )
                        # Execute on grid:
                        z1, ss1 = OK.execute("grid", self.longList[unNan_point], self.latiList[unNan_point])
                        for point, value, sigma2 in zip([nanList[count-1]], z1, ss1):
                            self.data.loc[i, point] = value
                    except ValueError:
                        for point in nanList:
                            self.data.loc[i, point] = r[point]

def rst(eTag):
    dc = DataConstants(eTag)
    cities, indicator, stations = dc.getPublicData()

    i = 0
    for city in cities:
        location_file = r"./data/"+city + "-station-location.csv"
        data_file = r'./data/' + eTag +'/'+city + '-' + indicator + '-masked.csv'
        lt_results = pd.read_csv(r'./results/' + eTag +'/'+city + '-' + indicator + '-lt-filling.csv')
        rst_results = r'./results/' + eTag +'/'+city + '-' + indicator + '-rst-filling.csv'
        data_masked = pd.read_csv(data_file)
        data_before_masked = pd.read_csv(r'./data/' + eTag +'/'+city + '-' + indicator + '-extracted.csv')

        rst = krigeAfterLinear_infill(location_file, data_file, stations[i], lt_results, data_masked, data_before_masked)
        rst.infill_krigeAfterLinear()
        rst.data.to_csv(rst_results, index=False)
        i=i+1
