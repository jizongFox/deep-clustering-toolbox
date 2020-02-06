__all__ = ["SemiDataSetInterface", "MedicalDatasetSemiInterface"]

from abc import abstractmethod
from copy import deepcopy as dcp
from itertools import repeat
from typing import Tuple, Callable, List, Type, Dict, Union

import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset

from deepclustering.augment import SequentialWrapper
from deepclustering.dataloader.dataset import CombineDataset
from deepclustering.dataset.segmentation import (
    MedicalImageSegmentationDataset,
    PatientSampler,
)
from deepclustering.decorator import FixRandomSeed


def _draw_indices(
    targets: np.ndarray,
    labeled_sample_num: int,
    class_nums: int = 10,
    validation_num: int = 5000,
    verbose: bool = True,
    seed: int = 1,
) -> Tuple[List[int], List[int], List[int]]:
    """
    draw indices for labeled and unlabeled dataset separations.
    :param targets: `torch.utils.data.Dataset.targets`-like numpy ndarray with all labels, used to split into labeled, unlabeled and validation dataset.
    :param labeled_sample_num: labeled sample number
    :param class_nums: num of classes in the target.
    :param validation_num: num of validation set, usually we split the big training set into `labeled`, `unlabeled`, `validation` sets, the `test` set is taken directly from the big test set.
    :param verbose: whether to print information while running.
    :param seed: random seed to draw indices
    :return: labeled indices and unlabeled indices
    """
    # todo add more rubost split method to enable fairness split of dataset, and validation set.
    labeled_sample_per_class = int(labeled_sample_num / class_nums)
    validation_sample_per_class = int(validation_num / class_nums) if class_nums else 0
    targets = np.array(targets)
    train_labeled_idxs: List[int] = []
    train_unlabeled_idxs: List[int] = []
    val_idxs: List[int] = []
    with FixRandomSeed(seed):
        for i in range(class_nums):
            idxs = np.where(targets == i)[0]
            np.random.shuffle(idxs)
            train_labeled_idxs.extend(idxs[:labeled_sample_per_class])
            train_unlabeled_idxs.extend(
                idxs[labeled_sample_per_class:-validation_sample_per_class]
            )
            val_idxs.extend(idxs[-validation_sample_per_class:])
        np.random.shuffle(train_labeled_idxs)
        np.random.shuffle(train_unlabeled_idxs)
        np.random.shuffle(val_idxs)
    if verbose:
        print(
            f">>>Generating {len(train_labeled_idxs)} labeled data, {len(train_unlabeled_idxs)} unlabeled data, and {len(val_idxs)} validation data."
        )
    assert train_labeled_idxs.__len__() + train_unlabeled_idxs.__len__() + len(
        val_idxs
    ) == len(targets), f"{1} split wrong."
    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


class SemiDataSetInterface:
    """
    Semi supervised dataloader creator interface
    """

    def __init__(
        self,
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
        verbose: bool = True,
    ) -> None:
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
            val_batch_size,
            verbose,
        )

        self.batch_params = {
            "labeled_batch_size": labeled_batch_size,
            "unlabeled_batch_size": unlabeled_batch_size,
            "val_batch_size": val_batch_size,
        }

        self.dataloader_params = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": drop_last,
        }

    def _init_labeled_unlabled_val_and_test_sets(
        self,
    ) -> Tuple[Subset, Subset, Subset, Dataset]:  # type: ignore
        """
        :param args: unknown args
        :param kwargs: unknown kwargs
        :return: Tuple of dataset, Labeled Dataset, Unlabeled Dataset, Val Dataset
        """
        train_set, test_set = self._init_train_test_sets()
        labeled_index, unlabeled_index, val_index = _draw_indices(
            train_set.targets,
            self.labeled_sample_num,
            class_nums=10,
            validation_num=5000,
            seed=self.seed,
            verbose=self.verbose,
        )
        # todo: to verify if here the dcp is necessary
        labeled_set = Subset(dcp(train_set), labeled_index)
        unlabeled_set = Subset(dcp(train_set), unlabeled_index)
        val_set = Subset(dcp(train_set), val_index)

        del train_set
        return labeled_set, unlabeled_set, val_set, test_set

    @staticmethod
    def _use_individual_batch_size(
        batch_size, l_batch_size, un_batch_size, val_batch_size, verbose
    ):
        if (
            isinstance(l_batch_size, int)
            and isinstance(un_batch_size, int)
            and isinstance(val_batch_size, int)
        ):
            assert (
                l_batch_size >= 1 and un_batch_size >= 1 and val_batch_size >= 1
            ), "batch_size should be greater than 1."
            if verbose:
                print(
                    f"Using labeled_batch_size={l_batch_size}, unlabeled_batch_size={un_batch_size}, val_batch_size={val_batch_size}"
                )
            return True
        elif isinstance(batch_size, int) and batch_size >= 1:
            if verbose:
                print(f"Using all same batch size of {batch_size}")
            return False
        else:
            raise ValueError(
                f"batch_size setting error, given batch_size={batch_size}, labeled_batch_size={l_batch_size}, "
                f"unlabeled_batch_size={un_batch_size}, val_batch_size={val_batch_size}."
            )

    @abstractmethod
    def _init_train_test_sets(self) -> Tuple[Dataset, Dataset]:
        raise NotImplementedError("train and test set initialization must be override")

    def _create_semi_supervised_datasets(
        self,
        labeled_transform: Callable[[Image.Image], Tensor],
        unlabeled_transform: Callable[[Image.Image], Tensor],
        val_transform: Callable[[Image.Image], Tensor],
        test_transform: Callable[[Image.Image], Tensor],
        target_transform: Callable[[Tensor], Tensor] = None,
    ) -> Tuple[Subset, Subset, Subset, Dataset]:
        (
            labeled_set,
            unlabeled_set,
            val_set,
            test_set,
        ) = self._init_labeled_unlabled_val_and_test_sets()
        labeled_set = self.override_transforms(
            labeled_set, labeled_transform, target_transform
        )
        unlabeled_set = self.override_transforms(
            unlabeled_set, unlabeled_transform, target_transform
        )
        val_set = self.override_transforms(val_set, val_transform, target_transform)
        test_set = self.override_transforms(test_set, test_transform, target_transform)
        return labeled_set, unlabeled_set, val_set, test_set

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
        test_transform: Callable[[Image.Image], Tensor],
        target_transform: Callable[[Tensor], Tensor] = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        _dataloader_params = dcp(self.dataloader_params)

        (
            labeled_set,
            unlabeled_set,
            val_set,
            test_set,
        ) = self._create_semi_supervised_datasets(
            labeled_transform=labeled_transform,
            unlabeled_transform=unlabeled_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            target_transform=target_transform,
        )
        if self._if_use_indiv_bz:
            _dataloader_params.update(
                {"batch_size": self.batch_params.get("labeled_batch_size")}
            )
        labeled_loader = DataLoader(labeled_set, **_dataloader_params)
        if self._if_use_indiv_bz:
            _dataloader_params.update(
                {"batch_size": self.batch_params.get("unlabeled_batch_size")}
            )
        unlabeled_loader = DataLoader(unlabeled_set, **_dataloader_params)
        _dataloader_params.update({"shuffle": False, "drop_last": False})
        if self._if_use_indiv_bz:
            _dataloader_params.update(
                {"batch_size": self.batch_params.get("val_batch_size")}
            )
        val_loader = DataLoader(val_set, **_dataloader_params)
        test_loader = DataLoader(test_set, **_dataloader_params)
        del _dataloader_params
        return labeled_loader, unlabeled_loader, val_loader, test_loader

    def SemiSupervisedParallelDataLoaders(
        self,
        labeled_transforms: List[Callable[[Image.Image], Tensor]],
        unlabeled_transforms: List[Callable[[Image.Image], Tensor]],
        val_transforms: List[Callable[[Image.Image], Tensor]],
        test_transforms: List[Callable[[Image.Image], Tensor]],
        target_transform: Callable[[Tensor], Tensor] = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:

        _dataloader_params = dcp(self.dataloader_params)

        def _override_transforms(dataset, img_transform_list, target_transform_list):
            # here deep copying the datasets are needed.
            return [
                self.override_transforms(dcp(dataset), img_trans, target_trans)
                for img_trans, target_trans in zip(
                    img_transform_list, target_transform_list
                )
            ]

        (
            labeled_set,
            unlabeled_set,
            val_set,
            test_set,
        ) = self._init_labeled_unlabled_val_and_test_sets()
        target_transform_list = repeat(target_transform)
        labeled_sets = _override_transforms(
            labeled_set, labeled_transforms, target_transform_list
        )
        unlabeled_sets = _override_transforms(
            unlabeled_set, unlabeled_transforms, target_transform_list
        )
        val_sets = _override_transforms(val_set, val_transforms, target_transform_list)
        test_sets = _override_transforms(
            test_set, test_transforms, target_transform_list
        )

        labeled_set = CombineDataset(*labeled_sets)
        unlabeled_set = CombineDataset(*unlabeled_sets)
        val_set = CombineDataset(*val_sets)
        test_set = CombineDataset(*test_sets)
        if self._if_use_indiv_bz:
            _dataloader_params.update(
                {"batch_size": self.batch_params.get("labeled_batch_size")}
            )
        labeled_loader = DataLoader(labeled_set, **_dataloader_params)
        if self._if_use_indiv_bz:
            _dataloader_params.update(
                {"batch_size": self.batch_params.get("unlabeled_batch_size")}
            )
        unlabeled_loader = DataLoader(unlabeled_set, **_dataloader_params)
        _dataloader_params.update({"shuffle": False, "drop_last": False})
        if self._if_use_indiv_bz:
            _dataloader_params.update(
                {"batch_size": self.batch_params.get("val_batch_size")}
            )
        val_loader = DataLoader(val_set, **_dataloader_params)
        test_loader = DataLoader(test_set, **_dataloader_params)
        return labeled_loader, unlabeled_loader, val_loader, test_loader


class MedicalDatasetSemiInterface:
    """
    Semi-supervised interface for datasets using `MedicalImageSegmentationDataset`
    """

    def __init__(
        self,
        DataClass: Type[MedicalImageSegmentationDataset],
        root_dir: str,
        labeled_data_ratio: float,
        unlabeled_data_ratio: float,
        seed: int = 0,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.DataClass = DataClass
        self.root_dir = root_dir
        assert (
            labeled_data_ratio + unlabeled_data_ratio
        ) <= 1, f"`labeled_data_ratio` + `unlabeled_data_ratio` should be less than 1.0, given {labeled_data_ratio + unlabeled_data_ratio}"
        self.labeled_ratio = labeled_data_ratio
        self.unlabeled_ratio = unlabeled_data_ratio
        self.val_ratio = 1 - (labeled_data_ratio + unlabeled_data_ratio)
        self.seed = seed
        self.verbose = verbose

    def compile_dataloader_params(
        self,
        batch_size: int = 1,
        labeled_batch_size: int = None,
        unlabeled_batch_size: int = None,
        val_batch_size: int = None,
        shuffle: bool = False,
        num_workers: int = 1,
        pin_memory: bool = True,
        drop_last=False,
    ):
        self._if_use_indiv_bz: bool = self._use_individual_batch_size(
            batch_size,
            labeled_batch_size,
            unlabeled_batch_size,
            val_batch_size,
            self.verbose,
        )
        if self._if_use_indiv_bz:
            self.batch_params = {
                "labeled_batch_size": labeled_batch_size,
                "unlabeled_batch_size": unlabeled_batch_size,
                "val_batch_size": val_batch_size,
            }
        self.dataloader_params = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": drop_last,
        }

    def SemiSupervisedDataLoaders(
        self,
        labeled_transform: SequentialWrapper = None,
        unlabeled_transform: SequentialWrapper = None,
        val_transform: SequentialWrapper = None,
        group_labeled=False,
        group_unlabeled=False,
        group_val=True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:

        _dataloader_params = dcp(self.dataloader_params)
        labeled_set, unlabeled_set, val_set = self._create_semi_supervised_datasets(
            labeled_transform=None, unlabeled_transform=None, val_transform=None
        )
        if labeled_transform is not None:
            labeled_set = self.override_transforms(labeled_set, labeled_transform)
        if unlabeled_transform is not None:
            unlabeled_set = self.override_transforms(unlabeled_set, unlabeled_transform)
        if val_transform is not None:
            val_set = self.override_transforms(val_set, val_transform)
        # labeled_dataloader
        if self._if_use_indiv_bz:
            _dataloader_params.update(
                {"batch_size": self.batch_params.get("labeled_batch_size")}
            )
        labeled_loader = (
            DataLoader(labeled_set, **_dataloader_params)
            if not group_labeled
            else self._grouped_dataloader(labeled_set, **_dataloader_params)
        )

        # unlabeled_dataloader
        if self._if_use_indiv_bz:
            _dataloader_params.update(
                {"batch_size": self.batch_params.get("unlabeled_batch_size")}
            )
        unlabeled_loader = (
            DataLoader(unlabeled_set, **_dataloader_params)
            if not group_unlabeled
            else self._grouped_dataloader(unlabeled_set, **_dataloader_params)
        )

        # val_dataloader
        _dataloader_params.update({"shuffle": False, "drop_last": False})
        if self._if_use_indiv_bz:
            _dataloader_params.update(
                {"batch_size": self.batch_params.get("val_batch_size")}
            )
        val_loader = (
            DataLoader(val_set, **_dataloader_params)
            if not group_val
            else self._grouped_dataloader(val_set, **_dataloader_params)
        )
        del _dataloader_params
        return labeled_loader, unlabeled_loader, val_loader

    @staticmethod
    def _use_individual_batch_size(
        batch_size, l_batch_size, un_batch_size, val_batch_size, verbose
    ) -> bool:
        if (
            isinstance(l_batch_size, int)
            and isinstance(un_batch_size, int)
            and isinstance(val_batch_size, int)
        ):
            assert (
                l_batch_size >= 1 and un_batch_size >= 1 and val_batch_size >= 1
            ), "batch_size should be greater than 1."
            if verbose:
                print(
                    f"Using labeled_batch_size={l_batch_size}, unlabeled_batch_size={un_batch_size}, val_batch_size={val_batch_size}"
                )
            return True
        elif isinstance(batch_size, int) and batch_size >= 1:
            if verbose:
                print(f"Using all same batch size of {batch_size}")
            return False
        else:
            raise ValueError(
                f"batch_size setting error, given batch_size={batch_size}, labeled_batch_size={l_batch_size}, "
                f"unlabeled_batch_size={un_batch_size}, val_batch_size={val_batch_size}."
            )

    def _create_semi_supervised_datasets(
        self,
        labeled_transform: SequentialWrapper = None,
        unlabeled_transform: SequentialWrapper = None,
        val_transform: SequentialWrapper = None,
    ) -> Tuple[
        MedicalImageSegmentationDataset,
        MedicalImageSegmentationDataset,
        MedicalImageSegmentationDataset,
    ]:
        raise NotImplementedError

    def _grouped_dataloader(
        self,
        dataset: MedicalImageSegmentationDataset,
        **dataloader_params: Dict[str, Union[int, float, bool]],
    ) -> DataLoader:
        """
        return a dataloader that requires to be grouped based on the reg of patient's pattern.
        :param dataset:
        :param shuffle:
        :return:
        """
        dataloader_params = dcp(dataloader_params)
        batch_sampler = PatientSampler(
            dataset=dataset,
            grp_regex=dataset._re_pattern,
            shuffle=dataloader_params.get("shuffle", False),
            verbose=self.verbose,
        )
        # having a batch_sampler cannot accept batch_size > 1
        dataloader_params["batch_size"] = 1
        dataloader_params["shuffle"] = False
        dataloader_params["drop_last"] = False
        return DataLoader(dataset, batch_sampler=batch_sampler, **dataloader_params)

    @staticmethod
    def override_transforms(
        dataset: MedicalImageSegmentationDataset, transform: SequentialWrapper
    ):
        assert isinstance(dataset, MedicalImageSegmentationDataset), dataset
        assert isinstance(transform, SequentialWrapper), transform
        dataset.set_transform(transform)
        return dataset
