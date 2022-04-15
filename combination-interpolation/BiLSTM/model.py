import torch.nn as nn
from BiLSTM.config import Config


class LSTM(nn.Module):
    """
    LSTM time series prediction layer and linear regression output layer
    """

    def __init__(self, config: Config):
        super().__init__()
        self.lstm = nn.LSTM(input_size=config.k, hidden_size=config.hidden_size,
                            num_layers=config.num_layer, batch_first=True, bidirectional=True)
        self.hid_linear = nn.Linear(in_features=config.input_size, out_features=1)
        self.mul_linear = nn.Linear(in_features=config.t*2+1, out_features=config.t*2+1)
        self.out_linear = nn.Linear(in_features=config.t*2+1, out_features=1)

    def forward(self, x):
        """
        x: [batch_size, seq_len, input_size]
        linear_out: [batch_size, seq_len, 1]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        lstm_out, _ = self.lstm(x)
        hid_linear_out = self.hid_linear(lstm_out)
        mul_linear_out = self.mul_linear(hid_linear_out.squeeze())
        linear_out = self.out_linear(mul_linear_out)
        return linear_out
