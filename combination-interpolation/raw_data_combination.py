import os
import pandas as pd

from DataConstants import DataConstants


def dataCombine(data_dir, selectedStations, eTag):
    # This program is used to merge the single files data downloaded from the Internet
    list = ['date', 'hour', 'type']
    dataColumns = []
    for l in selectedStations:
        l1 = list + l
        dataColumns.append(l1)

    dc = DataConstants(eTag)
    cities, indicator, stations = dc.getPublicData()

    i = 0
    for city in cities:
        df=[]
        for file in os.listdir(data_dir):
            data  = pd.read_csv(data_dir+file)
            df_temp = data.loc[data['type'] == indicator]
            df2_temp = df_temp[dataColumns[i]]
            df.append(df2_temp)

        result = pd.concat(df)
        result.to_csv('data/'+eTag+'/'+city+'-'+indicator+'-raw.csv',index=0)
        i = i + 1
