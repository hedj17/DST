import random
import pandas as pd
import numpy as np

from DataConstants import DataConstants


def extract_mask(value_list, probability, eTag):
    dc = DataConstants(eTag)
    cities, indicator, stations = dc.getPublicData()
    for city in cities:
        data0 = pd.read_csv('data/' + eTag + '/' + city + '-' + indicator + '-raw.csv')

        ls = []
        tr = data0.isnull().any(axis=1)
        for i in range(len(tr)):
            if (tr[i]):
                ls.append(i)
        df = []
        xulie = []
        for i in range(len(ls) - 1):
            if ls[i + 1] - ls[i] > 20:
                xulie.append((ls[i] + 1, ls[i + 1]))
                data1 = data0.iloc[ls[i] + 1:ls[i + 1]]
                df.append(data1)
        length = len(df)
        # print(df)
        # length represents the number of data blocks whose length is greater than 20
        data_end = pd.DataFrame(data=None, columns=data0.columns)

        for i in range(length):
            data_end = data_end.append(df[i])
        data_end.to_csv('data/' + eTag + '/' + city + '-' + indicator + '-extracted.csv', index=0)

        # The following code assigns np.nan where the value is "1" in the RAND matrix
        rand = []
        for p in range(length):
            data = df[p]
            m = data.shape[0]
            n = data.shape[1]
            data1 = np.random.random(size=(m, n))
            data1 = pd.DataFrame(data1)

            for i in range(m):
                for j in range(3, n):
                    if data1.iloc[i, j] < 0.04 and i > 2 and i < m - 2:
                        data1.iloc[i, j] = 1
                    else:
                        data1.iloc[i, j] = 0

            for i in range(m):
                for j in range(n):
                    if data1.iloc[i, j] == 1:
                        r = number_of_certain_probability(value_list, probability)
                        for k in range(r):
                            if i + k < m - 2:
                                data1.iloc[i + k, j] = 1
                            else:
                                break
            rand.append(data1)

        # The following code assigns np.nan where the value is "1" in the RAND matrix
        data_end = pd.DataFrame(data=None, columns=data0.columns)
        for p in range(length):
            data = df[p].copy()
            data1 = rand[p]
            m = data.shape[0]
            n = data.shape[1]
            for i in range(m):
                for j in range(3, n):
                    if data1.iloc[i, j] == 1:
                        data.iloc[i, j] = np.nan
            data_end = data_end.append(data)
        data_end.to_csv('data/' + eTag + '/' + city + '-' + indicator + '-masked.csv', index=0)


# this extract and mask method is ued to analyze the length of the column gap for different interpolation methods,
# i.e., e91, e93, e95 and e98
def extract_mask3(r, eTag):
    dc = DataConstants(eTag)
    cities, indicator, stations = dc.getPublicData()
    for city in cities:
        data0 = pd.read_csv('data/' + eTag + '/' + city + '-' + indicator + '-raw.csv')

        ls = []
        tr = data0.isnull().any(axis=1)
        for i in range(len(tr)):
            if tr[i]:
                ls.append(i)
        df = []
        xulie = []
        for i in range(len(ls) - 1):
            if ls[i + 1] - ls[i] > 30:
                xulie.append((ls[i] + 1, ls[i + 1]))
                data1 = data0.iloc[ls[i] + 1:ls[i + 1]]
                df.append(data1)
        length = len(df)
        # length represents the number of data blocks whose length is greater than 20
        data_end = pd.DataFrame(data=None, columns=data0.columns)

        for i in range(length):
            data_end = data_end.append(df[i])
        data_end.to_csv('data/' + eTag + '/' + city + '-' + indicator + '-extracted.csv', index=0)

        # The main purpose of the following code is to randomly set the value of the elements in the RAND matrix to "1"
        # , and rand is the list of random matrices
        rand = []
        for p in range(length):
            data = df[p]
            m = data.shape[0]
            n = data.shape[1]
            data1 = np.zeros((m, n))
            data1 = pd.DataFrame(data1)

            for j in range(3, n):
                count = 0
                for k in range(3):
                    count = count + 1
                    rr = random.randint(0, m - 1)
                    if data1.iloc[rr - 1, j] == 1:
                        continue
                    flag = False
                    for s in range(r + 1):
                        if rr + s < m:
                            if data1.iloc[rr + s, j] == 1:
                                flag = True
                                break
                        else:
                            flag = True
                            break
                    if flag is False:
                        for s in range(r):
                            data1.iloc[rr + s, j] = 1
            rand.append(data1)

        # The following code assigns np.nan where the value is "1" in the RAND matrix
        data_end = pd.DataFrame(data=None, columns=data0.columns)
        for p in range(length):
            data = df[p].copy()
            data1 = rand[p]
            m = data.shape[0]
            n = data.shape[1]
            for i in range(m):
                for j in range(3, n):
                    if data1.iloc[i, j] == 1:
                        data.iloc[i, j] = np.nan
            data_end = data_end.append(data)
        data_end.to_csv('data/' + eTag + '/' + city + '-' + indicator + '-masked.csv', index=0)


def number_of_certain_probability(sequence, probability):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(sequence, probability):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item
