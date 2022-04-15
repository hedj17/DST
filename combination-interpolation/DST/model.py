import torch.nn as nn
import torch


class CombineNet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=2, out_features=1, bias=True)

    def weight_init(self):
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        # x (batch_size)
        up_distance, down_distance, space_infill, sequential_infill, row_lack = \
            x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
        if space_infill is None:
            return sequential_infill

        #  rs and rt in formula 19
        row_lack = row_lack.unsqueeze(1)
        column_lack_new = (up_distance + down_distance - 1).unsqueeze(1)
        cat = torch.cat((row_lack, column_lack_new), dim=1)

        # calculate the weight
        weight_space = self.linear(cat.to(torch.float32))
        weight_space = torch.sigmoid(weight_space).squeeze()
        return weight_space * space_infill + (1 - weight_space) * sequential_infill


class CombineNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=2, out_features=1, bias=True)

    def weight_init(self):
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        # x (batch_size)
        up_distance, down_distance, space_infill, sequential_infill, row_lack = \
            x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
        if space_infill is None:
            return sequential_infill

        row_lack = row_lack.unsqueeze(1)
        column_lack_new = (torch.sqrt(up_distance ** 2 + down_distance ** 2) ** -1).unsqueeze(1)
        # column_lack_new = (torch.sqrt(up_distance*down_distance)/50).unsqueeze(1)
        cat = torch.cat((row_lack, column_lack_new), dim=1)

        weight_space = self.linear(cat.to(torch.float32))
        weight_space = torch.sigmoid(weight_space).squeeze()
        return weight_space * space_infill + (1 - weight_space) * sequential_infill


class CombineNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=2, out_features=1, bias=True)
        self.linear_2 = nn.Linear(in_features=2, out_features=1, bias=True)

    def weight_init(self):
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)

    def forward(self, x):
        # x (batch_size)
        up_distance, down_distance, space_infill, sequential_infill, row_lack = \
            x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
        if space_infill is None:
            return sequential_infill

        row_lack = row_lack.unsqueeze(1)
        column_lack_new = (up_distance + down_distance - 1).unsqueeze(1)
        # column_lack_new = (torch.sqrt(up_distance*down_distance)/50).unsqueeze(1)
        cat = torch.cat((row_lack, column_lack_new), dim=1)

        weight_space = self.linear_1(cat.to(torch.float32))
        weight_space = torch.sigmoid(weight_space).squeeze()

        weight_temp = self.linear_1(cat.to(torch.float32))
        weight_temp = torch.sigmoid(weight_temp).squeeze()
        return weight_space * space_infill + weight_temp * sequential_infill


class CombineNet4(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=2, out_features=1, bias=True)
        self.linear_2 = nn.Linear(in_features=2, out_features=1, bias=True)

    def weight_init(self):
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)

    def forward(self, x):
        # x (batch_size)
        up_distance, down_distance, space_infill, sequential_infill, row_lack = \
            x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
        if space_infill is None:
            return sequential_infill

        row_lack = row_lack.unsqueeze(1)
        column_lack_new = (torch.sqrt(up_distance ** 2 + down_distance ** 2) ** -1).unsqueeze(1)
        # column_lack_new = (torch.sqrt(up_distance*down_distance)/50).unsqueeze(1)
        cat = torch.cat((row_lack, column_lack_new), dim=1)

        # Calculate the weights of temporal interpolation and spatial interpolation respectively
        weight_space = self.linear_1(cat.to(torch.float32))
        weight_space = torch.sigmoid(weight_space).squeeze()

        weight_temp = self.linear_1(cat.to(torch.float32))
        weight_temp = torch.sigmoid(weight_temp).squeeze()
        return weight_space * space_infill + weight_temp * sequential_infill
