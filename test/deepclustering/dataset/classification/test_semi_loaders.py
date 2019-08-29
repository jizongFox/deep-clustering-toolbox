from unittest import TestCase

from deepclustering import DATA_PATH
from deepclustering.dataset.classification.cifar_helper import (
    Cifar10SemiSupervisedDatasetInterface,
    default_cifar10_img_transform,
    Cifar10SemiSupervisedParallelDatasetInterface
)
from deepclustering.dataset.classification.mnist_helper import (
    default_mnist_img_transform,
    MNISTSemiSupervisedDatasetInterface,
)


class Test_semisupervised_CIFAR(TestCase):
    def test_cifar_split(self):
        dataHandler = Cifar10SemiSupervisedDatasetInterface(
            data_root=DATA_PATH,
            labeled_sample_num=4000,
            img_transformation=default_cifar10_img_transform["tf1"],
        )
        labeled_loader, unlabeled_loader, val_loader = dataHandler.SemiSupervisedDataLoaders(
            batch_size=20, shuffle=True, num_workers=4, drop_last=False
        )
        assert (
                labeled_loader.dataset.__len__()
                + unlabeled_loader.dataset.__len__()
                + val_loader.dataset.__len__()
                == 60000
        )
        iter(labeled_loader).__next__()
        iter(unlabeled_loader).__next__()
        iter(val_loader).__next__()

    def test_mnist_split(self):
        dataHandler = MNISTSemiSupervisedDatasetInterface(
            data_root=DATA_PATH,
            labeled_sample_num=100,
            img_transformation=default_mnist_img_transform["tf1"],
        )
        labeled_loader, unlabeled_loader, val_loader = dataHandler.SemiSupervisedDataLoaders(
            batch_size=20, shuffle=True, num_workers=4, drop_last=False
        )
        assert (
                labeled_loader.dataset.__len__()
                + unlabeled_loader.dataset.__len__()
                + val_loader.dataset.__len__()
                == 70000
        )
        iter(labeled_loader).__next__()
        iter(unlabeled_loader).__next__()
        iter(val_loader).__next__()


class Test_cifar10SemiSupervisedParallelDatasetInterface(TestCase):
    def setUp(self) -> None:
        self.semisupervised_handler = Cifar10SemiSupervisedParallelDatasetInterface(data_root=DATA_PATH, batch_size=100,
                                                                                    shuffle=True)
        self.labeled_transform = default_cifar10_img_transform["tf1"]
        self.unlabeled_transform = default_cifar10_img_transform["tf2"]
        self.val_transform = default_cifar10_img_transform["tf3"]

    def test_cifar10(self):
        labeled_dataloader, unlabeled_dataloader, val_dataloader = self.semisupervised_handler.SemiSupervisedDataLoaders(
            labeled_transform=self.labeled_transform,
            unlabeled_transform=self.unlabeled_transform,
            val_transform=self.val_transform,
            target_transform=None
        )

    def test_cifar10_parallel(self):
        labeled_dataloader, unlabeled_dataloader, val_dataloader = self.semisupervised_handler.SemiSupervisedParallelDataLoaders(
            labeled_transforms=[self.labeled_transform] * 5,
            unlabeled_transforms=[self.unlabeled_transform] * 5,
            val_transforms=[self.val_transform],
            target_transform=None
        )
        print()