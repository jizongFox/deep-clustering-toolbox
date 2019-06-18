#
#
#  This is to test the semi supervised data loaders
#
#
__author__ = "Jizong Peng"
from unittest import TestCase

from deepclustering.dataset.classification.cifar_helper import Cifar10SemiSupervisedDatasetInterface, \
    default_cifar10_img_transform
from deepclustering.dataset.classification.mnist_helper import default_mnist_img_transform, \
    MNISTSemiSupervisedDatasetInterface


class Test_semisupervised_CIFAR(TestCase):

    def test_cifar_split(self):
        dataHandler = Cifar10SemiSupervisedDatasetInterface(
            data_root='/home/jizong/Workspace/deep-clustering-toolbox/.data',
            labeled_sample_num=4000,
            img_transformation=default_cifar10_img_transform['tf1'],
        )
        labeled_loader, unlabeled_loader, val_loader = dataHandler.SemiSupervisedDataLoaders(batch_size=20,
                                                                                             shuffle=True,
                                                                                             num_workers=4,
                                                                                             drop_last=False)
        assert labeled_loader.dataset.__len__() + unlabeled_loader.dataset.__len__() + val_loader.dataset.__len__() == 60000
        iter(labeled_loader).__next__()
        iter(unlabeled_loader).__next__()
        iter(val_loader).__next__()

    def test_mnist_split(self):
        dataHandler = MNISTSemiSupervisedDatasetInterface(
            data_root='/home/jizong/Workspace/deep-clustering-toolbox/.data',
            labeled_sample_num=100,
            img_transformation=default_mnist_img_transform['tf1'],
        )
        labeled_loader, unlabeled_loader, val_loader = dataHandler.SemiSupervisedDataLoaders(batch_size=20,
                                                                                             shuffle=True,
                                                                                             num_workers=4,
                                                                                             drop_last=False)
        assert labeled_loader.dataset.__len__() + unlabeled_loader.dataset.__len__() + val_loader.dataset.__len__() == 70000
        iter(labeled_loader).__next__()
        iter(unlabeled_loader).__next__()
        iter(val_loader).__next__()
