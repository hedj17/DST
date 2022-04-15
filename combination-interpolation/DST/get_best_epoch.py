import os
import torch
from sklearn.metrics import mean_squared_error
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error

model = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# select model
if model == 1:
    model_dir = r"./param1/"
else:
    model_dir = r"./param2/"


def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100


# calculation of SMAPE, RMSE, MAE
def get_error(loader, net, error):
    ty_lis = []
    y_lis = []
    for (x, y) in loader:
        ty = net(x.to(device)).to(device)
        for i, j in zip(ty.cpu().detach().numpy(), y.cpu().detach().numpy()):
            ty_lis.append(i)
            y_lis.append(j)
    ty_lis = np.array(ty_lis)
    y_lis = np.array(y_lis)
    if error == 0:
        m_error = mean_squared_error(ty_lis, y_lis, squared=False)
    elif error == 1:
        m_error = mean_absolute_error(ty_lis, y_lis)
    else:
        m_error = smape(y_lis, ty_lis)
    return m_error, ty_lis

#
# Import the weight file, calculate the RMSE of test set and training set respectively, and draw the image
# def main():
#     device = torch.device("cpu")
    error_dic = {}
    # 用每隔150次迭代时的网络模型计算相应训练数据和测试数据对应的RMSE值
    for file in os.listdir(model_dir):
        net = torch.load(model_dir + file + '', map_location="cpu")
        train_error = get_error(train_loader, net, 0)
        test_error = get_error(test_loader, net, 0)
        error = (train_error, test_error)
        # error_dic stores the RMSE values corresponding to the corresponding training data
        # and test data calculated by the network model corresponding to different iteration times
        error_dic.update({int(file.split('_')[0]): error})


    final_error = sorted(error_dic.items(), key=lambda item: item[0])
    print(min(final_error, key=lambda item: item[1][1]))
    x = [i[0] for i in final_error]
    y_train = [i[1][0] for i in final_error]
    y_test = [i[1][1] for i in final_error]
    plt.plot(x, y_train, marker='o', markersize=3, label='train')
    plt.plot(x, y_test, marker='o', markersize=3, label='test')
    plt.legend()
    plt.title('getting best epoch of TSI_{}'.format(model))
    plt.xlabel('epochs')
    #plt.xticks(np.linspace(0, 500, 11))
    plt.ylabel('value of RMSE')
    plt.show()

