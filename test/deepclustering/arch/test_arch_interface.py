from unittest import TestCase
from deepclustering.utils import simplex
import torch
from deepclustering import arch
from torch import nn


class Test_arch_interface(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.find_archs = {k: v for k, v in arch.__dict__.items() if type(v) == type}
        self.archs = {k: v for k, v in self.find_archs.items() if issubclass(v, nn.Module)}
        self.image = torch.randn(1, 3, 96, 96)

    def test_find_arch(self):
        net1 = arch.ClusterNet5g(**arch.ClusterNet5g_Param)
        pred = net1(self.image)
        print(pred.__len__())
        print(pred[0].shape)
        assert simplex(pred[0])

    def test_find_arch2(self):
        net2 = arch.ClusterNet5gTwoHead(**arch.ClusterNet5gTwoHead_Param)
        pred = net2(self.image)
        print(pred.__len__())
        print(pred[0].shape)
        assert simplex(pred[0])