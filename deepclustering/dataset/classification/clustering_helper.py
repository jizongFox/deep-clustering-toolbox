from abc import abstractmethod
from itertools import repeat
from typing import *

from torch.utils.data import Dataset, DataLoader

from .cifar import CIFAR10
from .mnist import MNIST
from .stl10 import STL10
from .. import dataset

__doc__ = "This interface is to define clustering datasets with different transformations"


class ClusterDatasetInterface(object):
    """
    dataset interface for unsupervised learning with combined train and test sets.
    """
    ALLOWED_SPLIT = []

    def __init__(self, DataClass: Dataset, split_partitions: List[str], batch_size: int = 1, shuffle: bool = False,
                 num_workers: int = 1, pin_memory: bool = True) -> None:
        """
        :param batch_size: batch_size = 1
        :param shuffle: shuffle the dataset, default = False
        :param num_workers: default 1
        """
        super().__init__()
        assert DataClass in (MNIST, CIFAR10, STL10), f"" \
            f"Dataset supported only by MNIST, CIFAR10 and STL-10, given{DataClass}."
        self.DataClass = DataClass
        if not isinstance(split_partitions, list):
            split_partitions = [split_partitions]
        assert isinstance(split_partitions[0], str), f"Elements of split_partitions must be str, " \
            f"given {split_partitions[0]}"
        self.split_partitions = split_partitions
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    @abstractmethod
    def _creat_concatDataset(self, image_transform: Callable, target_transform: Callable, dataset_dict: dict = {}):
        """
        create concat dataset with only one type of transform.
        :rtype: dataset
        :param image_transform:
        :param target_transform:
        :param dataset_dict:
        :return:
        """
        raise NotImplementedError

    def SerialDataLoader(self, image_transform: Callable = None, target_transform: Callable = None,
                         dataset_dict: Dict[str, Any] = {}, dataloader_dict: Dict[str, Any] = {}) -> DataLoader:
        r"""
        Combine several dataset in a serial way.
        :param image_transform: Callable function for both tran and val
        :param target_transform: Callable function for target such as remapping
        :param dataset_dict: supplementary options for datasets
        :param dataloader_dict: supplementary options for dataloader
        :return: type: Dataloader
        """
        concatSet = self._creat_concatDataset(image_transform, target_transform, dataset_dict)
        concatLoader = DataLoader(concatSet, batch_size=self.batch_size, shuffle=self.shuffle,
                                  num_workers=self.num_workers, drop_last=True, pin_memory=self.pin_memory,
                                  **dataloader_dict)
        return concatLoader

    def _creat_combineDataset(self, image_transforms: Tuple[Callable, ...], target_transform: Callable = None,
                              dataset_dict: Dict[str, Any] = {}):
        assert len(image_transforms) >= 1, f"Given {image_transforms}"
        assert not isinstance(target_transform,
                              (list, tuple)), f"We consider the target_transform should be the same for all."
        concatSets = []
        for t_img, t_tar in zip(image_transforms, repeat(target_transform)):
            concatSets.append(
                self._creat_concatDataset(image_transform=t_img, target_transform=t_tar, dataset_dict=dataset_dict))
        combineSet = dataset.CombineDataset(*concatSets)
        return combineSet

    def ParallelDataLoader(self, *image_transforms: Callable, target_transform: Callable = None,
                           dataset_dict: Dict[str, Any] = {}, dataloader_dict: Dict[str, Any] = {}) -> DataLoader:
        parallel_set = self._creat_combineDataset(image_transforms, target_transform, dataset_dict)
        parallel_loader = DataLoader(parallel_set, batch_size=self.batch_size, shuffle=self.shuffle,
                                     num_workers=self.num_workers, drop_last=True, pin_memory=self.pin_memory,
                                     **dataloader_dict)
        return parallel_loader
