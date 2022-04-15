import pandas as pd
from DataConstants import DataConstants


def lt(eTag):
    dc = DataConstants(eTag)
    cities, indicator, stations = dc.getPublicData()

    for city in cities:
        data_masked = pd.read_csv(r'./data/'+ eTag +'/'+city + '-' + indicator + '-masked.csv')
        results = data_masked.interpolate(method='linear', limit_direction='both')
        infilled_file = "./results/" + eTag +'/'+city + "-" + indicator + "-lt-filling.csv"

        results.to_csv(infilled_file, index=False)
