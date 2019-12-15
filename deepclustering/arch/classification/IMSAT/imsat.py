"""
This file contains the network for IMSAT paper: https://arxiv.org/pdf/1702.08720.pdf.
The code is taken from https://github.com/MOhammedJAbi/Imsat/blob/master/Imsat.py
"""
import math
from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from ....utils import _warnings


class IMSATNet(nn.Module):
    def __init__(
        self,
        in_channel: int = 784,
        output_k_A: int = 50,
        output_k_B: int = 10,
        num_sub_heads: int = 5,
        *args,
        **kwargs
    ):
        _warnings(args, kwargs)
        super(IMSATNet, self).__init__()
        self.fc1 = nn.Linear(in_channel, 1200)
        torch.nn.init.normal_(self.fc1.weight, std=0.1 * math.sqrt(2 / (28 * 28)))
        self.fc1.bias.data.fill_(0)
        self.fc2 = nn.Linear(1200, 1200)
        torch.nn.init.normal_(self.fc2.weight, std=0.1 * math.sqrt(2 / 1200))
        self.fc2.bias.data.fill_(0)
        self.bn1 = nn.BatchNorm1d(1200, eps=2e-5)
        self.bn1_F = nn.BatchNorm1d(1200, eps=2e-5, affine=False)
        self.bn2 = nn.BatchNorm1d(1200, eps=2e-5)
        self.bn2_F = nn.BatchNorm1d(1200, eps=2e-5, affine=False)
        self.head_A = IMSATHeader(output_k=output_k_A, num_sub_heads=num_sub_heads)
        self.head_B = IMSATHeader(output_k=output_k_B, num_sub_heads=num_sub_heads)

    def forward(self, x, head="B", *args, **kwargs) -> List[torch.Tensor]:
        """
        output gives the logit value
        :param x:
        :param head: the head to perform inference
        :param update_batch_stats:
        :return:
        """
        if x.shape.__len__() == 4:
            x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        if head == "B":
            x = self.head_B(x)
        elif head == "A":
            x = self.head_A(x)
        return x


class IMSATHeader(nn.Module):
    def __init__(self, output_k=10, num_sub_heads=5):
        super().__init__()
        self.output_k = output_k
        self.num_sub_heads = num_sub_heads

        self.heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(1200, self.output_k), nn.Softmax(dim=1))
                for _ in range(self.num_sub_heads)
            ]
        )

    def forward(self, input):
        results = []
        for i in range(self.num_sub_heads):
            results.append(self.heads[i](input))
        return results


IMSATNet_Param = {"output_k_A": 50, "output_k_B": 10, "num_sub_heads": 5}
