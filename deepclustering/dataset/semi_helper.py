#
# This is the semi-supervised meta class for mnist, cifar and svhn
#
#
from abc import abstractmethod
from typing import Tuple, Callable, List, Union, Any, Dict

from numpy.random import choice
from torch.utils.data import Dataset, DataLoader, Subset

from deepclustering.dataset.DataLoader_helper import BackgroundGenerator
from ..dataset import CombineDataset
from deepclustering.decorator import FixRandomSeed
from deepclustering.utils import _warnings


class SemiDatasetInterface(object):
    """
    Dataset interface for semi supervised learning, which generates indices for samples.
    """

    def __init__(
            self,
            DataClass: Dataset,
            data_root: str,
            labeled_sample_num: int,
            img_transformation: Callable = None,
            target_transformation: Callable = None,
            *args,
            **kwargs,
    ) -> None:
        super().__init__()
        _warnings(args, kwargs)
        assert isinstance(labeled_sample_num, int)
        self.data_root = data_root
        self.DataClass = DataClass
        self.labeled_sample_num = labeled_sample_num
        self.img_transform = img_transformation
        self.target_transform = target_transformation

    @abstractmethod
    def _init_train_and_test_test(
            self, transform, target_transform, *args, **kwargs
    ) -> Tuple[Dataset, Dataset]:
        """
        This method initialize the train set and validation set
        :return:
        """
        _warnings(args, kwargs)

    def SemiSupervisedDataLoaders(
            self,
            batch_size=4,
            shuffle=True,
            drop_last=False,
            num_workers=1,
            *args,
            **kwargs,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        _warnings(args, kwargs)
        train_set, val_set = self._init_train_and_test_test(
            transform=self.img_transform, target_transform=self.target_transform
        )
        labeled_index, unlabeled_index = self._draw_indices(
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

    @staticmethod
    def _draw_indices(
            dataset: Dataset, labeled_sample_num: int, verbose: bool = True
    ) -> Tuple[List[int], List[int]]:
        total_num = len(dataset)
        labeled_indices = sorted(
            choice(list(range(total_num)), labeled_sample_num, replace=False)
        )
        unlabeled_indices = sorted(list(set(range(total_num)) - set(labeled_indices)))
        if verbose:
            print(f">>>Generating {len(labeled_indices)} labeled data and {len(unlabeled_indices)} unlabeled data.")
        return labeled_indices, unlabeled_indices


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

    @staticmethod
    def _draw_indices(
            dataset: Dataset, labeled_sample_num: int, verbose: bool = True
    ) -> Tuple[List[int], List[int]]:
        total_num = len(dataset)
        labeled_indices = sorted(choice(list(range(total_num)), labeled_sample_num, replace=False))
        unlabeled_indices = sorted(list(set(range(total_num)) - set(labeled_indices)))
        if verbose:
            print(f">>>Generating {len(labeled_indices)} labeled data and {len(unlabeled_indices)} unlabeled data.")
        return labeled_indices, unlabeled_indices

    def _create_semi_supervised_dataset(self, image_transform: Callable, target_transform: Callable) -> Tuple[
        Dataset, Dataset, Dataset]:
        train_set, val_set = self._init_train_and_test_test(
            transform=image_transform, target_transform=target_transform
        )
        with FixRandomSeed(1):
            labeled_index, unlabeled_index = self._draw_indices(
                train_set, self.labeled_sample_num
            )
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
