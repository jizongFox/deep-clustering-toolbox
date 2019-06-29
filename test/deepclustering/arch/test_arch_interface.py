from unittest import TestCase

import torch
from deepclustering import arch
from deepclustering.arch import get_arch, ARCH_CALLABLES, ARCH_PARAM_DICT
from deepclustering.utils import simplex
from torch import nn


class Test_arch_interface(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.find_archs = {k: v for k, v in arch.__dict__.items() if type(v) == type}
        self.archs = {
            k: v for k, v in self.find_archs.items() if issubclass(v, nn.Module)
        }
        self.image = torch.randn(1, 3, 64, 64)

    #
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

    def test_find_arch3(self):
        net2 = arch.ClusterNet6c(**arch.ClusterNet6c_Param)
        pred = net2(self.image)
        print(pred.__len__())
        print(pred[0].shape)
        assert simplex(pred[0])

    def test_find_arch4(self):
        net2 = arch.ClusterNet6cTwoHead(**arch.ClusterNet6cTwoHead_Param)
        pred = net2(self.image)
        print(pred.__len__())
        print(pred[0].shape)
        assert simplex(pred[0])

    def test_interface(self):
        net_keys = ARCH_CALLABLES.keys()
        for k in net_keys:
            print(f"Building network {k}...")
            net = get_arch(k, ARCH_PARAM_DICT[k])
            if k == "clusternetimsat":
                with self.assertRaises(RuntimeError):
                    pred = net(self.image)
            else:
                pred = net(self.image)
            print(pred.__len__())
            print(pred[0].shape)
            # assert simplex(pred[0])
