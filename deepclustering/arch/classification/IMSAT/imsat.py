"""
This file contains the network for IMSAT paper: https://arxiv.org/pdf/1702.08720.pdf.
The code is taken from https://github.com/MOhammedJAbi/Imsat/blob/master/Imsat.py
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class IMSATNet(nn.Module):
    def __init__(self):
        super(IMSATNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1200)
        torch.nn.init.normal_(self.fc1.weight, std=0.1 * math.sqrt(2 / (28 * 28)))
        self.fc1.bias.data.fill_(0)
        self.fc2 = nn.Linear(1200, 1200)
        torch.nn.init.normal_(self.fc2.weight, std=0.1 * math.sqrt(2 / 1200))
        self.fc2.bias.data.fill_(0)
        self.fc3 = nn.Linear(1200, 10)
        torch.nn.init.normal_(self.fc3.weight, std=0.0001 * math.sqrt(2 / 1200))
        self.fc3.bias.data.fill_(0)
        self.bn1 = nn.BatchNorm1d(1200, eps=2e-5)
        self.bn1_F = nn.BatchNorm1d(1200, eps=2e-5, affine=False)
        self.bn2 = nn.BatchNorm1d(1200, eps=2e-5)
        self.bn2_F = nn.BatchNorm1d(1200, eps=2e-5, affine=False)

    def forward(self, x, update_batch_stats=True):
        """
        output gives the logit value
        :param x:
        :param update_batch_stats:
        :return:
        """
        if x.shape.__len__()==4:
            x = x.view(x.size(0),-1)
        if not update_batch_stats:
            x = self.fc1(x)
            x = self.bn1_F(x) * self.bn1.weight + self.bn1.bias
            x = F.relu(x)
            x = self.fc2(x)
            x = self.bn2_F(x) * self.bn2.weight + self.bn2.bias
            x = F.relu(x)
            x = self.fc3(x)
            return x
        else:
            x = self.fc1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.fc3(x)
            return x


IMSATNet_Param = {}
