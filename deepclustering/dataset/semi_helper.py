# This file is the abstract class for semi-supervised learning for classification.
# It might not be applicable for semi-supervised segmentation where you have groups of images defined by their natures
from abc import abstractmethod
from copy import deepcopy as dcp
from itertools import repeat
from typing import Tuple, Callable, List, Type

from PIL import Image
from numpy.random import choice
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset

from deepclustering.decorator import FixRandomSeed
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
    # todo add more rubost split method to enable fairness split of dataset, and validation set.
    total_num = len(dataset)
    targets = dataset.targets
    assert total_num >= labeled_sample_num, f"`labeled_sample_num={labeled_sample_num} should be smaller than totoal_num={total_num}.`"
    with FixRandomSeed(seed):
        # only fix numpy and random pkgs
        labeled_indices = sorted(choice(list(range(total_num)), labeled_sample_num, replace=False))
    unlabeled_indices = sorted(list(set(range(total_num)) - set(labeled_indices)))
    if verbose:
        print(f">>>Generating {len(labeled_indices)} labeled data and {len(unlabeled_indices)} unlabeled data.")
    assert labeled_indices.__len__() + unlabeled_indices.__len__() == total_num, f"{1} split wrong."
    return labeled_indices, unlabeled_indices


class SemiDataSetInterface(object):

    def __init__(self,
                 DataClass: Type[Dataset],
                 data_root: str,
                 labeled_sample_num: int,
                 seed: int = 0,
                 batch_size: int = 1,
                 labeled_batch_size: int = None,
                 unlabeled_batch_size: int = None,
                 val_batch_size: int = None,
                 shuffle: bool = False,
                 num_workers: int = 1,
                 pin_memory: bool = True,
                 drop_last=False,
                 verbose: bool = True) -> None:
        """
        when batch_size is not `None`, we do not consider `labeled_batch_size`, `unlabeled_batch_size`, and `val_batch_size`
        when batch_size is `None`, `labeled_batch_size`,`unlabeled_batch_size` and `val_batch_size` should be all int and >=1
        """
        super().__init__()
        self.data_root = data_root
        self.DataClass = DataClass
        self.seed = seed
        self.labeled_sample_num = labeled_sample_num
        self.verbose = verbose
        self._if_use_indiv_bz: bool = self._use_individual_batch_size(
            batch_size,
            labeled_batch_size,
            unlabeled_batch_size,
            val_batch_size, verbose)

        self.batch_params = {
            "labeled_batch_size": labeled_batch_size,
            "unlabeled_batch_size": unlabeled_batch_size,
            "val_batch_size": val_batch_size
        }

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
        labeled_index, unlabeled_index = _draw_indices(train_set, self.labeled_sample_num, seed=self.seed,
                                                       verbose=self.verbose)
        # todo: to verify if here the dcp is necessary
        labeled_set = Subset(dcp(train_set), labeled_index)
        unlabeled_set = Subset(dcp(train_set), unlabeled_index)

        del train_set
        return labeled_set, unlabeled_set, val_set

    @staticmethod
    def _use_individual_batch_size(batch_size, l_batch_size, un_batch_size, val_batch_size, verbose):
        if isinstance(l_batch_size, int) and isinstance(un_batch_size, int) and isinstance(val_batch_size, int):
            assert l_batch_size >= 1 and un_batch_size >= 1 and val_batch_size >= 1, "batch_size should be greater than 1."
            if verbose:
                print(
                    f"Using labeled_batch_size={l_batch_size}, unlabeled_batch_size={un_batch_size}, val_batch_size={val_batch_size}")
            return True
        elif isinstance(batch_size, int) and batch_size >= 1:
            if verbose:
                print(f"Using all same batch size of {batch_size}")
            return False
        else:
            raise ValueError(
                f"batch_size setting error, given batch_size={batch_size}, labeled_batch_size={l_batch_size}, "
                f"unlabeled_batch_size={un_batch_size}, val_batch_size={val_batch_size}.")

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
            target_transform: Callable[[Tensor], Tensor] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        _dataloader_params = dcp(self.dataloader_params)

        labeled_set, unlabeled_set, val_set = self._create_semi_supervised_datasets(
            labeled_transform,
            unlabeled_transform,
            val_transform,
            target_transform)
        if self._if_use_indiv_bz:
            _dataloader_params.update({"batch_size": self.batch_params.get("labeled_batch_size")})
        labeled_loader = DataLoader(labeled_set, **_dataloader_params)
        if self._if_use_indiv_bz:
            _dataloader_params.update({"batch_size": self.batch_params.get("unlabeled_batch_size")})
        unlabeled_loader = DataLoader(unlabeled_set, **_dataloader_params)
        _dataloader_params.update({"shuffle": False})
        if self._if_use_indiv_bz:
            _dataloader_params.update({"batch_size": self.batch_params.get("val_batch_size")})
        val_loader = DataLoader(val_set, **_dataloader_params)
        return labeled_loader, unlabeled_loader, val_loader

    def SemiSupervisedParallelDataLoaders(
            self,
            labeled_transforms: List[Callable[[Image.Image], Tensor]],
            unlabeled_transforms: List[Callable[[Image.Image], Tensor]],
            val_transforms: List[Callable[[Image.Image], Tensor]],
            target_transform: Callable[[Tensor], Tensor] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        _dataloader_params = dcp(self.dataloader_params)

        def _override_transforms(dataset, img_transform_list, target_transform_list):
            # here deep copying the datasets are needed.
            return [self.override_transforms(dcp(dataset), img_trans, target_trans) for img_trans, target_trans in
                    zip(img_transform_list, target_transform_list)]

        labeled_set, unlabeled_set, val_set = self._init_labeled_unlabled_val_sets()
        target_transform_list = repeat(target_transform)
        labeled_sets = _override_transforms(labeled_set, labeled_transforms, target_transform_list)
        unlabeled_sets = _override_transforms(unlabeled_set, unlabeled_transforms, target_transform_list)
        val_sets = _override_transforms(val_set, val_transforms, target_transform_list)

        labeled_set = CombineDataset(*labeled_sets)
        unlabeled_set = CombineDataset(*unlabeled_sets)
        val_set = CombineDataset(*val_sets)
        labeled_loader = DataLoader(labeled_set, **_dataloader_params)
        unlabeled_loader = DataLoader(unlabeled_set, **_dataloader_params)
        _dataloader_params.update({"shuffle": False})
        val_loader = DataLoader(val_set, **_dataloader_params)
        return labeled_loader, unlabeled_loader, val_loader
