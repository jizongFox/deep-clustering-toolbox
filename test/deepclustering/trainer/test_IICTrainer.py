from unittest import TestCase

import torch
from deepclustering.dataset import Cifar10ClusteringDataloaders, default_cifar10_img_transform
from deepclustering.model import Model
from deepclustering.trainer.IIC_trainer import IICTrainer


class TestIICTrainer(TestCase):
    def setUp(self) -> None:
        self.arch_dict = {'name': 'clusternet5g', 'input_size': 32, 'num_channel': 1, 'output_k': 10,
                          'num_sub_heads': 2}
        self.optim_dict = {'name': 'Adam', 'lr': 0.005}
        self.scheduler_dict = {'name': 'MultiStepLR', 'milestones': [10, 20, 30, 40, 50, 60, 70, 80, 90], 'gamma': 0.7}
        self.trainer_dict = {'max_epoch': 3, 'device': 'cuda'}
        self.dataloader_dict = {'batch_size': 64, 'shuffle': True, "num_workers": 8}
        self.train_dataloader = Cifar10ClusteringDataloaders(**self.dataloader_dict).creat_CombineDataLoader(
            default_cifar10_img_transform['tf1'],
            default_cifar10_img_transform['tf2'])
        self.val_dataloader = Cifar10ClusteringDataloaders(
            **{**self.dataloader_dict, **{"shuffle": False}}).creat_CombineDataLoader(
            default_cifar10_img_transform['tf3'])
        self.model = Model(self.arch_dict, self.optim_dict, self.scheduler_dict)

        self.IICtrainer = IICTrainer(model=self.model, train_loader=self.train_dataloader,
                                     val_loader=self.val_dataloader, **self.trainer_dict)

    def test__train_loop(self):
        self.IICtrainer._train_loop(self.train_dataloader, 0)

    def test__eval_loop(self):
        with torch.no_grad():
            self.IICtrainer._eval_loop(self.val_dataloader, 0)

    def test_starttraining(self):
        self.IICtrainer.start_training()
