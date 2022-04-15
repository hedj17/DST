import pandas as pd
import numpy as np

# upper missing value, lower missing value, row missing value, column missing value
from DataConstants import DataConstants


def feature_statistic(dataType, eTag):
    dc = DataConstants(eTag)
    cities, indicator, stations = dc.getPublicData()

    for city in cities:
        data_end = pd.read_csv('data/'+eTag+'/'+city + '-' + indicator + '-' + dataType + '.csv')
        # print(data_end.iloc[0:1,:])
        m = data_end.shape[0]
        n = data_end.shape[1]
        tr1 = data_end.isnull()
        up_null = np.random.random(size=(m, n))
        up_null = pd.DataFrame(up_null)
        down_null = np.random.random(size=(m, n))
        down_null = pd.DataFrame(down_null)
        column_null = np.random.random(size=(m, n))
        column_null = pd.DataFrame(column_null)
        up_null.iloc[:, 0:3] = data_end.iloc[:, 0:3]
        down_null.iloc[:, 0:3] = data_end.iloc[:, 0:3]
        column_null.iloc[:, 0:3] = data_end.iloc[:, 0:3]

        for j in range(3, n):
            i = 0
            while i < m:
                if tr1.iloc[i, j]:
                    k = i + 1
                    while k < m and tr1.iloc[k, j]:
                        k += 1
                    d = k - i
                    for e in range(d):
                        up_null.iloc[e + i, j] = e + 1
                        down_null.iloc[e + i, j] = d - e
                        column_null.iloc[e + i, j] = d
                    i = k
                else:
                    up_null.iloc[i, j] = 0
                    down_null.iloc[i, j] = 0
                    column_null.iloc[i, j] = 0
                    i += 1

        up_null.to_csv('data/'+eTag+'/'+city + '-' + indicator + '-upNUll.csv', index=0)
        down_null.to_csv('data/'+eTag+'/'+city + '-' + indicator + '-upNull.csv', index=0)
        column_null.to_csv('data/'+eTag+'/'+city + '-' + indicator + '-columnNull.csv', index=0)

        row_null = np.random.random(size=(m, n))
        row_null = pd.DataFrame(row_null)
        for i in range(3):
            row_null.iloc[:, i] = data_end.iloc[:, i]
        for i in range(m):
            j = 0
            d = 0
            for j in range(3, n):
                if tr1.iloc[i, j]:
                    d += 1
            for j in range(3, n):
                if tr1.iloc[i, j]:
                    row_null.iloc[i, j] = d
                else:
                    row_null.iloc[i, j] = 0
        row_null.to_csv('data/'+eTag+'/'+city + '-' + indicator + '-rowNull.csv', index=0)

        dataset = iter([up_null, down_null, row_null, column_null])
        data = []

        null = data_end

        for item in dataset:
            data.append([item.iloc[i, j] for i, j in zip(np.where(pd.isnull(null))[0], np.where(pd.isnull(null))[
                1])])
        data = np.array(data).T

        dataD = pd.DataFrame(data)
        dataD.to_csv('data/'+ eTag +'/'+city + '-' + indicator + '-nullFeature-' + dataType + '.csv', index=0)
        print(city, " is done for feature statistics!")