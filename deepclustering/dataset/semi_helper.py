from abc import abstractmethod
from copy import deepcopy as dcp
from itertools import repeat
from pathlib import Path
from typing import Tuple, Callable, List, Union, Any, Dict

import numpy as np
from PIL import Image
from deepclustering.decorator import FixRandomSeed
from deepclustering.utils import _warnings
from numpy.random import choice
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset

from ..dataloader.dataset import CombineDataset


def _draw_indices(dataset: Dataset, labeled_sample_num: int, verbose: bool = True, seed: int = 1) \
        -> Tuple[List[int], List[int]]:
    """
    draw indices for labeled and unlabeled dataset separations.
    :param dataset: `torch.utils.data.Dataset`-like dataset, used to split into labeled and unlabeled dataset.
    :param labeled_sample_num: labeled sample number
    :param verbose: whether to print information while running.
    :param seed: random seed to draw indices
    :return: labeled indices and unlabeled indices
    """
    total_num = len(dataset)
    assert total_num >= labeled_sample_num, f"`labeled_sample_num={labeled_sample_num} should be smaller than totoal_num={total_num}.`"
    with FixRandomSeed(seed):
        # only fix numpy and random pkgs
        labeled_indices = sorted(choice(list(range(total_num)), labeled_sample_num, replace=False))
    unlabeled_indices = sorted(list(set(range(total_num)) - set(labeled_indices)))
    if verbose:
        print(f">>>Generating {len(labeled_indices)} labeled data and {len(unlabeled_indices)} unlabeled data.")
    assert labeled_indices.__len__() + unlabeled_indices.__len__() == total_num, f"{1} split wrong."
    return labeled_indices, unlabeled_indices


class SemiDatasetInterface(object):
    """
    Dataset interface for semi supervised learning, which generates indices for samples.
    """

    def __init__(
            self,
            DataClass: Dataset,
            data_root: str,
            labeled_sample_num: int,
            img_transformation: Callable[[Image.Image], Tensor] = None,
            target_transformation: Callable[[Union[Tensor, np.ndarray]], Tensor] = None,
            *args,
            **kwargs,
    ) -> None:
        super(SemiDatasetInterface, self).__init__()
        _warnings(args, kwargs)
        assert isinstance(labeled_sample_num, int), "`labeled_sample_num` is expected to be `int`, given {}".format(
            type(labeled_sample_num))
        assert isinstance(data_root, (str, Path)), "`data_root` is expected to be 'str-like', given {}".format(
            type(data_root))
        self.data_root = data_root
        self.DataClass = DataClass
        self.labeled_sample_num = labeled_sample_num
        self.img_transform = img_transformation
        self.target_transform = target_transformation

    @abstractmethod
    def _init_train_and_test_test(
            self, transform, target_transform, *args, **kwargs
    ) -> Tuple[Dataset, Dataset]:  # type: ignore
        """
        This method initialize the train set and validation set
        :return: train and test dataset
        """
        _warnings(args, kwargs)

    def SemiSupervisedDataLoaders(self, batch_size=4,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=1, *args, **kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
        _warnings(args, kwargs)
        train_set, val_set = self._init_train_and_test_test(
            transform=self.img_transform, target_transform=self.target_transform
        )
        labeled_index, unlabeled_index = _draw_indices(
            train_set, self.labeled_sample_num
        )
        labeled_loader = DataLoader(
            Subset(train_set, labeled_index),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            *args,
            **kwargs,
        )
        unlabeled_loader = DataLoader(
            Subset(train_set, unlabeled_index),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            *args,
            **kwargs,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            *args,
            **kwargs,
        )

        assert (labeled_loader.dataset.__len__() + unlabeled_loader.dataset.__len__() == train_set.__len__())

        return labeled_loader, unlabeled_loader, val_loader


class SemiParallelDatasetInterface(object):
    # todo: to be completed in the future

    def __init__(
            self,
            DataClass: Dataset,
            data_root: str,
            labeled_sample_num: int,
            batch_size: int = 1,
            shuffle: bool = False,
            num_workers: int = 1,
            pin_memory: bool = True,
            drop_last=False,
    ) -> None:
        super().__init__()
        self.DataClass = DataClass
        self.data_root = data_root
        self.labeled_sample_num = labeled_sample_num
        self.dataloader_params = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": drop_last
        }

    @abstractmethod
    def _init_train_and_test_test(
            self, transform, target_transform, *args, **kwargs
    ) -> Tuple[Dataset, Dataset]:
        pass

    def _create_semi_supervised_datasets(self, image_transform: Callable, target_transform: Callable) -> Tuple[
        Dataset, Dataset, Dataset]:
        train_set, val_set = self._init_train_and_test_test(
            transform=image_transform, target_transform=target_transform
        )

        labeled_index, unlabeled_index = _draw_indices(train_set, self.labeled_sample_num)
        labeled_set, unlabeled_set = Subset(train_set, labeled_index), Subset(train_set, unlabeled_index)
        return labeled_set, unlabeled_set, val_set

    def _creat_combineDataset(self) -> CombineDataset:
        pass

    def ParallelDataLoaders(
            self,
            *image_transforms: Callable,
            target_transform: Union[Callable, Tuple[Callable, ...]] = None,
            dataloader_dict: Dict[str, Any] = {}
    ):
        pass


class SemiDataSetInterface_(object):

    def __init__(self,
                 DataClass: Dataset,
                 data_root: str,
                 labeled_sample_num: int,
                 seed: int = 0,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 num_workers: int = 1,
                 pin_memory: bool = True,
                 drop_last=False) -> None:
        super().__init__()
        self.data_root = data_root
        self.DataClass = DataClass
        self.seed = seed
        self.labeled_sample_num = labeled_sample_num
        self.dataloader_params = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": drop_last
        }

    def _init_labeled_unlabled_val_sets(self) -> Tuple[Subset, Subset, Dataset]:  # type: ignore
        """
        :param args: unknown args
        :param kwargs: unknown kwargs
        :return: Tuple of dataset, Labeled Dataset, Unlabeled Dataset, Val Dataset
        """
        train_set, val_set = self._init_train_val_sets()
        labeled_index, unlabeled_index = _draw_indices(train_set, self.labeled_sample_num, seed=self.seed)
        #
        labeled_set = Subset(dcp(train_set), labeled_index)

        unlabeled_set = Subset(dcp(train_set), unlabeled_index)

        del train_set
        return labeled_set, unlabeled_set, val_set

    @abstractmethod
    def _init_train_val_sets(self) -> Tuple[Dataset, Dataset]:
        raise NotImplementedError("train and test set initialization must be override")

    def _create_semi_supervised_datasets(
            self,
            labeled_transform: Callable[[Image.Image], Tensor],
            unlabeled_transform: Callable[[Image.Image], Tensor],
            val_transform: Callable[[Image.Image], Tensor],
            target_transform: Callable[[Tensor], Tensor] = None
    ) -> Tuple[Subset, Subset, Dataset]:
        labeled_set, unlabeled_set, val_set = self._init_labeled_unlabled_val_sets()
        labeled_set = self.override_transforms(labeled_set, labeled_transform, target_transform)
        unlabeled_set = self.override_transforms(unlabeled_set, unlabeled_transform, target_transform)
        val_set = self.override_transforms(val_set, val_transform, target_transform)
        return labeled_set, unlabeled_set, val_set

    @staticmethod
    def override_transforms(dataset, img_transform, target_transform):
        assert isinstance(dataset, (Dataset, Subset))
        if isinstance(dataset, Subset):
            dataset.dataset.transform = img_transform
            dataset.dataset.target_transform = target_transform
        else:
            dataset.transform = img_transform
            dataset.target_transform = target_transform
        return dataset

    def SemiSupervisedDataLoaders(
            self,
            labeled_transform: Callable[[Image.Image], Tensor],
            unlabeled_transform: Callable[[Image.Image], Tensor],
            val_transform: Callable[[Image.Image], Tensor],
            target_transform: Callable[[Image.Image], Tensor] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        labeled_set, unlabeled_set, val_set = self._create_semi_supervised_datasets(
            labeled_transform,
            unlabeled_transform,
            val_transform,
            target_transform)
        labeled_loader = DataLoader(labeled_set, **self.dataloader_params)
        unlabeled_loader = DataLoader(unlabeled_set, **self.dataloader_params)
        self.dataloader_params.update({"shuffle": False})
        val_loader = DataLoader(val_set, **self.dataloader_params)
        return labeled_loader, unlabeled_loader, val_loader

    def SemiSupervisedParallelDataLoaders(
            self,
            labeled_transforms: List[Callable[[Image.Image], Tensor]],
            unlabeled_transforms: List[Callable[[Image.Image], Tensor]],
            val_transforms: List[Callable[[Image.Image], Tensor]],
            target_transform: Callable[[Tensor], Tensor] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:

        def _override_transforms(dataset, img_transform_list, target_transform_list):
            return [self.override_transforms(dataset, img_trans, target_trans) for img_trans, target_trans in
                    zip(img_transform_list, target_transform_list)]

        labeled_set, unlabeled_set, val_set = self._init_labeled_unlabled_val_sets()
        target_transform_list = repeat(target_transform)

        labeled_sets = _override_transforms(labeled_set, labeled_transforms, target_transform_list)
        unlabeled_sets = _override_transforms(unlabeled_set, unlabeled_transforms, target_transform_list)
        val_sets = _override_transforms(val_set, val_transforms, target_transform_list)

        labeled_set = CombineDataset(*labeled_sets)
        unlabeled_set = CombineDataset(*unlabeled_sets)
        val_set = CombineDataset(*val_sets)
        labeled_loader = DataLoader(labeled_set, **self.dataloader_params)
        unlabeled_loader = DataLoader(unlabeled_set, **self.dataloader_params)
        self.dataloader_params.update({"shuffle": False})
        val_loader = DataLoader(val_set, **self.dataloader_params)
        return labeled_loader, unlabeled_loader, val_loader
