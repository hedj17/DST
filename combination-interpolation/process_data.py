import pandas as pd
from datetime import datetime


def datetime2str(dt):
    return str(dt.date()) + '-' + str(dt.hour)


def str2datatime(strdt):
    return datetime.strptime(strdt, '%Y-%m-%d-%H')


def str2index(strdt):
    y, m, d, h = strdt.split('-')
    return y + m + d, h

def checkNan(row, stations):
    # input pd.Series
    NanList = []
    UnNanList = []
    for index in stations:
        # index = str(i) + 'A'
        tt = row[index]
        if pd.isnull(tt):
            NanList.append(index)
        else:
            UnNanList.append(index)
    return NanList, UnNanList


class process:
    def __init__(self, df):
        df['date'] = df['date'].astype('object')
        df['hour'] = df['hour'].astype('object')
        self.data = df

    def read_data(self):
        self.data.drop(['Unnamed: 0', 'type'], axis=1, inplace=True)
        for index, row in self.data.iterrows():
            date = str(row['date'])
            hour = str(row['hour'])
            date_now = datetime.datetime.strptime(date + hour, '%Y%m%d%H')
            self.data.iloc[index, 0] = date_now
        self.data.drop('hour', axis=1, inplace=True)
        return self.data

    def time_window(self, seq_len, target_size):
        data = self.read_data()
        print(data)
        if len(data) <= seq_len + target_size:
            raise Exception(
                "The Length of data is %d, while affect by (window_size = %d).".format(len(data),
                                                                                       seq_len + target_size))
        df = pd.DataFrame()
        for i in range(seq_len):
            df['x{}'.format(i)] = data.values.tolist()[i:-(target_size + seq_len - i)]
        for t in range(target_size):
            print(seq_len, -(target_size - t))
            df['y{}'.format(t)] = data.values.tolist()[seq_len + t:-(target_size - t)]

        return df


if __name__ == "__main__":
    data = pd.read_csv(r".\AQI.csv")
    print(process(data).read_data())
