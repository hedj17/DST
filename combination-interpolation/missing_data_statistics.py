import pandas as pd
import numpy as np

from DataConstants import DataConstants

def missingDataStatistic(eTag, dataType):
    dc = DataConstants(eTag)
    cities, indicator, stations = dc.getPublicData()

    for city in cities:
        data = pd.read_csv('data/'+ eTag +'/'+city + '-' + indicator + '-' + dataType +'.csv')

        results = data.isnull().sum()

        results[3:].to_csv('data/' + eTag +'/'+city + '-' + indicator + '-missingValue-statistics'+ '-' + dataType +'.csv')
        # print(city, ":", results)