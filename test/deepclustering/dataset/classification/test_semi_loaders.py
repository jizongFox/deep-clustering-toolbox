from unittest import TestCase

import torch
from deepclustering import DATA_PATH
from deepclustering.dataset.classification.cifar_helper import (
    default_cifar10_img_transform,
    Cifar10SemiSupervisedDatasetInterface
)
from deepclustering.dataset.classification.mnist_helper import (
    default_mnist_img_transform,
    MNISTSemiSupervisedDatasetInterface,
)


class Test_semisupervised_CIFAR(TestCase):
    def test_cifar_single_transform(self):
        dataHandler = Cifar10SemiSupervisedDatasetInterface(
            data_root=DATA_PATH,
            labeled_sample_num=4000,
            seed=1,
            batch_size=10,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        labeled_loader, unlabeled_loader, val_loader = dataHandler.SemiSupervisedDataLoaders(
            labeled_transform=default_cifar10_img_transform["tf1"],
            unlabeled_transform=default_cifar10_img_transform["tf2"],
            val_transform=default_cifar10_img_transform["tf3"],
            target_transform=None
        )
        assert (
                labeled_loader.dataset.__len__()
                + unlabeled_loader.dataset.__len__()
                + val_loader.dataset.__len__()
                == 60000
        )
        imgs, targets = iter(labeled_loader).__next__()
        assert imgs.shape == torch.Size([10, 1, 32, 32])
        imgs, targets = iter(unlabeled_loader).__next__()
        assert imgs.shape == torch.Size([10, 1, 32, 32])
        imgs, targets = iter(val_loader).__next__()
        assert imgs.shape == torch.Size([10, 1, 32, 32])

    def test_cifar_different_batch_size(self):
        dataHandler = Cifar10SemiSupervisedDatasetInterface(
            data_root=DATA_PATH,
            labeled_sample_num=4000,
            seed=1,
            labeled_batch_size=10,
            unlabeled_batch_size=20,
            val_batch_size=30,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        labeled_loader, unlabeled_loader, val_loader = dataHandler.SemiSupervisedDataLoaders(
            labeled_transform=default_cifar10_img_transform["tf1"],
            unlabeled_transform=default_cifar10_img_transform["tf2"],
            val_transform=default_cifar10_img_transform["tf3"],
            target_transform=None
        )
        imgs, targets = iter(labeled_loader).__next__()
        assert imgs.shape == torch.Size([10, 1, 32, 32])
        imgs, targets = iter(unlabeled_loader).__next__()
        assert imgs.shape == torch.Size([20, 1, 32, 32])
        imgs, targets = iter(val_loader).__next__()
        assert imgs.shape == torch.Size([30, 1, 32, 32])

    def test_cifar10_parallel_loader(self):
        dataHandler = Cifar10SemiSupervisedDatasetInterface(
            data_root=DATA_PATH,
            labeled_sample_num=4000,
            seed=1,
            batch_size=10,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        labeled_loader, unlabeled_loader, val_loader = dataHandler.SemiSupervisedParallelDataLoaders(
            labeled_transforms=[default_cifar10_img_transform["tf1"]] * 5,
            unlabeled_transforms=[default_cifar10_img_transform["tf2"]] * 2,
            val_transforms=[default_cifar10_img_transform["tf3"]] * 1,
            target_transform=None
        )
        assert (
                labeled_loader.dataset.__len__()
                + unlabeled_loader.dataset.__len__()
                + val_loader.dataset.__len__()
                == 60000
        )
        labeled_imgs, labeled_targets = zip(*iter(labeled_loader).__next__())
        assert len(labeled_imgs) == 5
        assert labeled_imgs[0].shape == torch.Size([10, 1, 32, 32])


class Test_semisupervised_MNIST(TestCase):
    def test_mnist_single_transform(self):
        dataHandler = MNISTSemiSupervisedDatasetInterface(batch_size=100)
        labeled_loader, unlabeled_loader, val_loader = dataHandler.SemiSupervisedDataLoaders(
            labeled_transform=default_mnist_img_transform["tf1"],
            unlabeled_transform=default_mnist_img_transform["tf2"],
            val_transform=default_mnist_img_transform["tf3"],
            target_transform=None
        )
        assert (
                labeled_loader.dataset.__len__()
                + unlabeled_loader.dataset.__len__()
                + val_loader.dataset.__len__()
                == 70000
        )
        imgs, targts = iter(labeled_loader).__next__()
        assert imgs.shape == torch.Size([100, 1, 24, 24])

    def test_mnist_parallel_transform(self):
        dataHandler = MNISTSemiSupervisedDatasetInterface(batch_size=100)
        labeled_loader, unlabeled_loader, val_loader = dataHandler.SemiSupervisedParallelDataLoaders(
            labeled_transforms=[default_mnist_img_transform["tf1"]] * 5,
            unlabeled_transforms=[default_mnist_img_transform["tf2"]] * 4,
            val_transforms=[default_mnist_img_transform["tf3"]] * 3,
            target_transform=None
        )
        assert (
                labeled_loader.dataset.__len__()
                + unlabeled_loader.dataset.__len__()
                + val_loader.dataset.__len__()
                == 70000
        )
        imgs, targts = zip(*iter(labeled_loader).__next__())
        assert len(imgs) == 5
        assert imgs[0].shape == torch.Size([100, 1, 24, 24])
