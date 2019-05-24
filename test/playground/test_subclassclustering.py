"""
This file contains the subclass clustering test file.
"""
import warnings
from unittest import TestCase

import torch

from deepclustering import DATA_PATH
from deepclustering.augment.tensor_augment import Resize
from deepclustering.dataset.classification.mnist import MNIST
from deepclustering.model import Model
from playground.subspaceClustering.subclassClustering import SubSpaceClusteringMethod


class testSubspaceClusteringMethod(TestCase):
    def setUp(self) -> None:
        super().setUp()
        # take a minibatch of mnist to set an example.
        mnist_dataset = MNIST(root=DATA_PATH, train=False)
        self.imgs = mnist_dataset.data[:200].unsqueeze(1).float()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.imgs = Resize((32, 32), interpolation='bilinear')(self.imgs).float()
        assert self.imgs.shape == torch.Size([200, 1, 32, 32])
        # for the model initialization
        arch_param = {
            'name': 'clusternet5g',
            'num_channel': 1,
            'batchnorm_track': True
        }
        optim_dict = {
            'name': 'Adam',
            'lr': 0.0001,
            'weight_decay': 1e-4
        }
        scheduler_dict = {
            'name': 'MultiStepLR',
            'milestones': [100, 200, 300, 400, 500, 600, 700, 800, 900],
            'gamma': 0.75
        }
        model = Model(arch_param, optim_dict, scheduler_dict)
        self.subspaceMethod = SubSpaceClusteringMethod(model=model, device=torch.device('cpu'))

    def test_MainInterface(self):
        index = torch.linspace(0, 20, 20).long()
        self.subspaceMethod.set_input(imgs=self.imgs[index], index=index)
        self.subspaceMethod.update()
