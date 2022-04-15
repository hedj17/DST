import numpy as np
import pandas as pd
from DataConstants import DataConstants


def get_distance(sites_location: pd.DataFrame):
    """get distance of two sites"""
    sites_location = sites_location.set_index("监测点编码")
    n = sites_location.shape[0]
    distance = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                lng1 = sites_location["经度"][i]
                lng2 = sites_location["经度"][j]
                lat1 = sites_location["纬度"][i]
                lat2 = sites_location["纬度"][j]
                distance[i][j] = ((lng1 - lng2) ** 2 + (lat1 - lat2) ** 2) ** 0.5
                distance[i][j] = distance[i][j]
    return distance


def interpolation(data_masked, eTag, city, k, c):
    m = data_masked.shape[0]
    n = data_masked.shape[1]

    null = pd.isnull(data_masked)

    # the furthest station is removed in the experiment 'e84'
    if eTag in ['e84', 'e86']:
        distance = get_distance(pd.read_csv(r"./data/"+ city +"-station-location-selected.csv"))
    else:
        distance = get_distance(pd.read_csv(r"./data/"+ city +"-station-location.csv"))

    results = data_masked
    for i in range(1, m - 1):
        for j in range(n):
            if null.iloc[i, j] == True:
                cnt = 0
                for e in range(n):
                    for p in range(-2, 2):
                        if null.iloc[i + p, e] == False:
                            cnt += data_masked.iloc[i + p, e] / (k * (distance[j][e] ** 2) + p ** 2 + c)
                results.iloc[i][j] = cnt
    for j in range(n):
        if null.iloc[0, j] == True:
            cnt = 0
            for e in range(n):
                for p in range(2):
                    if null.iloc[p, e] == False:
                        cnt += data_masked.iloc[p, e] / (k * (distance[j][e] ** 2) + p ** 2 + c)
            results.iloc[0][j] = cnt
    for j in range(n):
        if null.iloc[m - 1, j] == True:
            cnt = 0
            for e in range(n):
                for p in range(m - 2, m):
                    if null.iloc[p, e] == False:
                        cnt += data_masked.iloc[p, e] / (k * (distance[j][e] ** 2) + (m - 1 - p) ** 2 + c)
            results.iloc[m - 1][j] = cnt

    return results


def efim(eTag):
    dc = DataConstants(eTag)
    cities, indicator, stations = dc.getPublicData()

    for city in cities:
        data = pd.read_csv(r'./data/' + eTag + '/' + city + '-' + indicator + '-masked.csv')
        first_three_column = data.iloc[:, 0:3]

        if eTag == 'e84':
            if city == 'xian':
                c = 37
                k = 1
            else:
                c=20
                k=14
        elif eTag == 'e86':
            if city == 'xian':
                c = 38
                k = 13
            else:
                c=19
                k=27
        else:
            if city == 'xian' and indicator == 'PM2.5':
                c = 40
                k = 14
            if city == 'chengdu' and indicator == 'PM2.5':
                c = 23
                k = 13
            if city == 'xian' and indicator == 'PM10':
                c = 39
                k = 21
            if city == 'chengdu' and indicator == 'PM10':
                c = 23
                k = 1
        data_masked = data.iloc[:, 3:]
        results = interpolation(data_masked, eTag, city, k, c)
        final_results = first_three_column.join(results)
        infilled_file = "./results/" + eTag + '/' + city + "-" + indicator + "-efim-filling.csv"

        final_results.to_csv(infilled_file, index=False)
