import os.path

import torch


class Config:
    def __init__(self):
        # model
        self.device = 'cpu' if torch.cuda.is_available() else 'cpu'
        self.hidden_size = 64  #
        self.num_layer = 3  #
        self.bidirectional = True
        self.input_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.out_feature = 1

        # train
        self.lr = 0.0001
        self.epoch = 30
        self.weight_decay = 0.01
        self.batch_size = 128
        self.clip = 1

        # loader
        self.k = 3
        self.t = 4
        self.raw_path = "data/xian-PM10-raw.csv"
        self.site_path = "data/xian-station-location.csv"
        self.param_root_path = "BiLSTM/cache"
        self.model_path = 'BiLSTM/cache/LSTM_epoch198_TrainLoss6.6422_TestLoss2.4834'
        self.train_loader_path = os.path.join(self.param_root_path, "train_loader.pkl")
        self.test_loader_path = os.path.join(self.param_root_path, "test_loader.pkl")
        self.scaler_path = os.path.join(self.param_root_path, "scaler.pkl")


config = Config()
