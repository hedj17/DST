import pandas as pd
import numpy as np

data = pd.read_csv('data/e80/xian-PM2.5-masked.csv')

m = data.shape[0]
n = data.shape[1]
print(m)
print(n)

dic = {}

for j in range(3, n):
    flag = False
    count = 0
    for i in range(m):
        if np.isnan(data.iloc[i, j]):
            if flag is False:
                flag = True
            count = count + 1
        else:
            if flag is True:
                flag = False
                if count not in dic.keys():
                    dic[count] = 1
                else:
                    dic[count] = dic[count] + 1
                count = 0

        if flag is True and i == m-1:
            flag = False
            if count not in dic.keys():
                dic[count] = 1
            else:
                dic[count] = dic[count] + 1
            count = 0

print(dic)

final_dic = sorted(dic.items(), key=lambda item: item[0])

print(final_dic)

# pd.DataFrame(final_dic).to_csv('xian_gap_count.csv', index=False)