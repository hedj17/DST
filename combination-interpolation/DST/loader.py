import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class Loader():
    def __init__(self, city, indicator, eTag):
        # up_distance.csv: Record the distance of each missing data point from the
        # last nearest non missing value in the time dimension, that is, the missing value
        # dtub in Formula 20
        self.up_null = pd.read_csv(r'./data/' + eTag + '/' + city + '-' + indicator + '-upNull.csv')

        # down_distance.csv: Record the distance from each missing data point to the next nearest non missing value
        # in the time dimension, that is, the missing value
        # dtlb in Formula 20
        self.down_null = pd.read_csv(r'./data/' + eTag + '/' + city + '-' + indicator + '-downNull.csv')

        # line.csv: The interpolation result of linear interpolation method is added on the basis of AQI_raw.csv
        self.liner_null = pd.read_csv(r'./results/' + eTag + '/' + city + '-' + indicator + '-lt-filling.csv')

        # kri.csv:The interpolation result of ordinary Kriging interpolation method is added on the basis of AQI_raw.csv
        self.kri_null = pd.read_csv(r'./results/' + eTag + '/' + city + '-' + indicator + '-ok-filling.csv')

        # row_null.csv: Record the number of missing values of each missing data point in the spatial dimension,
        # that is, the missing value of the line
        # dtlb in Formula 19
        self.row_null = pd.read_csv(r'./data/' + eTag + '/' + city + '-' + indicator + '-rowNull.csv')

        # AQI_null.csv：For AQI_raw.csv After artificial data masking
        self.data_file = pd.read_csv(r'./data/' + eTag + '/' + city + '-' + indicator + '-masked.csv')

        # AQI-raw.csv：A collection of 20 consecutive rows or more of data blocks
        # that do not contain missing values filtered from online download data
        self.data_before_masked = pd.read_csv(r'./data/' + eTag + '/' + city + '-' + indicator + '-extracted.csv')

    def train_test_loader(self, percentage, batch_size):
        dataset = iter(
            [self.up_null, self.down_null, self.liner_null, self.kri_null, self.row_null])
        data = []
        row_indexs = []
        column_indexs = []

        self.data_file.iloc[:, 0:3] = -1

        for item in dataset:
            data.append([item.iloc[int(i), int(j)] for i, j in
                         zip(np.where(pd.isnull(self.data_file))[0], np.where(pd.isnull(self.data_file))[
                             1])])
            # i. J is the row number and column number corresponding to each missing value

        for i, j in zip(np.where(pd.isnull(self.data_file))[0], np.where(pd.isnull(self.data_file))[1]):
            row_indexs.append(i)
            column_indexs.append(j)

        data1 = np.vstack((np.array(data), np.array(row_indexs)))

        data2 = np.vstack((np.array(data1), np.array(column_indexs)))

        data3 = data2.T

        # data = np.array(data).T
        # Data is the data of n * m, n is the number of missing values,
        # and M is 5, which respectively represents the dtub,
        # lower missing value, linear interpolation result,
        # Kriging interpolation result and row missing value of each missing value
        target = np.array([self.data_before_masked.iloc[i, j] for i, j in
                           zip(np.where(pd.isnull(self.data_file))[0], np.where(pd.isnull(self.data_file))[1])])

        # target: The real value corresponding to each artificial missing value
        X_train1, X_test1, y_train, y_test = train_test_split(data3, target, test_size=percentage, random_state=0)

        X_train = X_train1[:, 0:5]
        X_test = X_test1[:, 0:5]
        X_test_row_index = X_test1[:, 5]
        X_test_column_index = X_test1[:, 6]

        train_loader = DataLoader(MySet(X_train, y_train), batch_size=batch_size, shuffle=False)

        test_loader = DataLoader(MySet(X_test, y_test), batch_size=batch_size, shuffle=False)

        return train_loader, test_loader, X_train, X_test, y_train, y_test, X_test_row_index, X_test_column_index


class MySet(Dataset):
    def __init__(self, data_set, data_target):
        self.up = data_set[:, 0]
        self.down = data_set[:, 1]
        self.lin = data_set[:, 2]
        self.kri = data_set[:, 3]
        self.row = data_set[:, 4]
        self.y = data_target

    def __getitem__(self, index):
        return torch.tensor([self.up[index], self.down[index], self.kri[index], self.lin[index], self.row[index]]), \
               self.y[index]

    def __len__(self):
        return len(self.up)


if __name__ == '__main__':
    cities = ["xian", "chengdu", "urumqi", "yinchuan"]
    learning_rate = 0.001
    indicator = 'PM2.5'
    model = 2

    for city in cities:
        loader = Loader(city, indicator)
        train_loader, test_loader, _, _, _, _, _, _ = loader.train_test_loader(0.3, 128)
        break
