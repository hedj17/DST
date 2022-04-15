from typing import List

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from BiLSTM.model import LSTM
import torch
from torch import nn
from BiLSTM.config import Config



def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight)
    if hasattr(m, 'bias') and hasattr(m.bias, 'fill_'):
        nn.init.constant_(m.bias, 0)


class Test:
    def __init__(self):
        self.eval_func_dict = {
            'smape': lambda x, y: 2.0 * np.mean(np.abs(x - y) / (np.abs(x) + np.abs(y))) * 100,
            'mse': lambda x, y: mean_squared_error(x, y),
            'rmse': lambda x, y: np.sqrt(mean_squared_error(x, y)),
            'mae': lambda x, y: mean_absolute_error(x, y)
        }

    # pred_func: nast.forward
    def evaluate(self, model, test_loader, scaler, test_category: List[str] = None):
        if test_category is None:
            test_category = ['smape', 'mse', 'rmse', 'mae']
        ty_lis = []
        y_lis = []
        for x, y in test_loader:
            output = model(x)
            for predict_y, true_y in zip(output.cpu().detach(), y.detach()):
                ty_lis.append(float(true_y))  # bs, tf
                y_lis.append(float(predict_y))

        return {
            c: self.eval_func_dict[c](scaler.inverse_transform(np.array(ty_lis)),
                                      scaler.inverse_transform(np.array(y_lis))) for c in test_category
        }


def train(config: Config, train_loader, test_loader, scaler):
    criterion = nn.MSELoss()
    model = LSTM(config).to(config.device)
    model.apply(initialize_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_model_loss = float("inf")
    best_model_sd = None
    test_model = Test()
    for epoch in range(config.epoch):
        train_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            infill = model(x)
            loss = criterion(infill.squeeze(), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            optimizer.step()
            train_loss += loss.item()
        print("Epoch{}: Training_loss = {}".format(epoch + 1, train_loss))

        test_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                output = model(x)
                loss = criterion(y, output.squeeze())
                test_loss += loss.item()
        print("Epoch{}: Test_loss = {}".format(epoch + 1, test_loss))
        print('eval', test_model.evaluate(model, test_loader, scaler=scaler))

        # save model
        if best_model_loss > test_loss:
            best_model_loss = test_loss
            best_model_sd = model.state_dict()
    return best_model_sd