from unittest import TestCase

import torch
from deepclustering.arch import ARCH_PARAM_DICT
from deepclustering.model import Model


class Test_Default_Module(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.arch_dicts = ARCH_PARAM_DICT
        self.arch_names = list(ARCH_PARAM_DICT.keys())
        for arch_name in self.arch_names:
            self.arch_dicts[arch_name].update({'name': arch_name})
            # Built the arch_dicts with the default arch_dict from the architecture.
        self.optim_dict = {
            'name': 'Adam',
            'lr': 1e-4,
            'weight_decay': 1e-5
        }
        self.img = torch.randn(10, 3, 64, 64)

    def test_no_argument(self):
        model = Model()
        model(self.img)

    def test_partial_argument(self):
        for k, arch_dict in self.arch_dicts.items():
            if k == 'clusternetimsat':
                with self.assertRaises(RuntimeError):
                    model = Model(arch_dict=arch_dict, optim_dict=self.optim_dict)
                    model(self.img)
                    model.schedulerStep()
            else:
                model = Model(arch_dict=arch_dict, optim_dict=self.optim_dict)
                try:
                    model(self.img)
                except:
                    model(self.img[:, 0].unsqueeze(1))
                model.schedulerStep()
