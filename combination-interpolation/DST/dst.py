import os
import shutil

import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from DST.get_best_epoch import get_error
from DST.loader import Loader
from DST.model import CombineNet1, CombineNet2, CombineNet3
from DataConstants import DataConstants


def train(city, eTag, train_loader, test_loader,  model, learning_rate, epochs):
    if model == 1:
        net = CombineNet1()
        save_model_dir = "./data/temp/cgdst/"+city+"-"+eTag+"/"
        save_num = int(epochs/50)
    elif model == 2:
        net = CombineNet2()
        save_model_dir = "./data/temp/fgdst/"+city+"-"+eTag+"/"
        save_num = int(epochs/50)
    elif model == 3:
        net = CombineNet3()
        save_model_dir = "./data/temp/cgdst1/"+city+"-"+eTag+"/"
        save_num = int(epochs/50)
    elif model == 4:
        net = CombineNet3()
        save_model_dir = "./data/temp/fgdst1/"+city+"-"+eTag+"/"
        save_num = int(epochs/50)

    isExists = os.path.exists(save_model_dir)
    if isExists:
        shutil.rmtree(save_model_dir)
        os.mkdir(save_model_dir)
    else:
        os.makedirs(save_model_dir)

    net.weight_init()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    print("\n", "##" * 10, "  NetWork  ", "##" * 10, "\n", net, "\n", "##" * 26, "\n")

    # define loss function and  optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-2)
    for epoch in range(epochs):
        running_loss = 0
        for inputs, true_value in train_loader:
            inputs, true_value = inputs.to(device), true_value.to(torch.float32).to(device)
            optimizer.zero_grad()
            combine_infill = net(inputs).to(torch.float32)
            loss = criterion(combine_infill, true_value)
            loss.backward()
            optimizer.step()
            running_loss += loss.data
        if (epoch + 1) % save_num == 0:
            torch.save(net,
                       save_model_dir + "{}_{:.3f}.pkl".format(epoch + 1, running_loss))

    # Analyze whether there are over fitting and under fitting problems in model training
    error_dic = {}
    # Use the network model every 150 iterations to calculate the RMSE value
    # corresponding to the corresponding training data and test data
    for file in os.listdir(save_model_dir):
        net = torch.load(save_model_dir + file + '')
        train_error, _ = get_error(train_loader, net, 0)
        test_error, _ = get_error(test_loader, net, 0)
        error = (train_error, test_error)

        # error_dic stores the RMSE values corresponding to the corresponding training data
        # and test data calculated by the network model corresponding to different iteration times
        error_dic.update({int(file.split('_')[0]): error})


    final_error = sorted(error_dic.items(), key=lambda item: item[0])

    x = [i[0] for i in final_error]
    y_train = [i[1][0] for i in final_error]
    y_test = [i[1][1] for i in final_error]

    #
    # plt.plot(x, y_train, marker='o', markersize=3, label='train')
    # plt.plot(x, y_test, marker='o', markersize=3, label='test')
    # plt.legend()
    # plt.title('getting best epoch of DST_{}'.format(model))
    # plt.xlabel('epochs')
    # #plt.xticks(np.linspace(0, 500, 11))
    # plt.ylabel('value of RMSE')
    #
    # if model == 1:
    #     fig = r'./results/' + eTag + '/' + city + '-cgdst-fitting.png'
    # if model == 2:
    #     fig = r'./results/' + eTag + '/' + city + '-fgdst-fitting.png'
    # else:
    #     fig = r'./results/' + eTag + '/' + city + '-biased-fitting.png'
    #
    # plt.savefig(fig)
    # plt.show()
    print("Train Done!")
    return net


def test(net, test_loader):
    rmse, estimatedValues = get_error(test_loader, net, 0)
    mae, estimatedValues = get_error(test_loader, net, 1)
    smape, estimatedValues = get_error(test_loader, net, 2)
    return rmse, mae, smape, estimatedValues


def dst(model, eTag, learning_rates, epochs):
    dc = DataConstants(eTag)
    cities, indicator, stations = dc.getPublicData()

    for city in cities:
        learning_rate=learning_rates[city]
        epoch = epochs[city]
        loader = Loader(city, indicator, eTag)
        train_loader, test_loader, _, _, _, _, X_test_row_index, X_test_column_index = loader.train_test_loader(0.3, 128)
        net = train(city, eTag, train_loader, test_loader, model, learning_rate, epoch)
        rmse, mae, smape, estimatedValues = test(net, test_loader)
        print("City:", city, "-----RMSE:", rmse, " MAE:", mae, " SMAPE:", smape)

        masked_data = pd.read_csv(r'./data/'+ eTag +'/'+city + '-' + indicator + '-masked.csv')

        for index in range(len(estimatedValues)):
            masked_data.iloc[int(X_test_row_index[index]),int(X_test_column_index[index])] = estimatedValues[index]

        if model == 1:
            dst_results_file = r'./results/' + eTag + '/' + city + '-' + indicator + '-cgdst-filling.csv'
        if model == 2:
            dst_results_file = r'./results/' + eTag + '/' + city + '-' + indicator + '-fgdst-filling.csv'
        if model == 3:
            dst_results_file = r'./results/' + eTag + '/' + city + '-' + indicator + '-cgdst1-filling.csv'
        if model == 4:
            dst_results_file = r'./results/' + eTag + '/' + city + '-' + indicator + '-fgdst1-filling.csv'

        masked_data.to_csv(dst_results_file, index=False)
