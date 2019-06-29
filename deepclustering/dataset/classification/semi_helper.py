#
# This is the semi-supervised meta class for mnist, cifar and svhn
#
#
from abc import abstractmethod
from typing import Tuple, Callable, List

from numpy.random import choice
from torch.utils.data import Dataset, DataLoader, Subset
from ..DataLoader_helper import BackgroundGenerator
from ...utils import _warnings


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
        verbose: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        _warnings(args, kwargs)
        assert isinstance(labeled_sample_num, int)
        self.DataClass = DataClass
        self.labeled_sample_num = labeled_sample_num
        self.img_transform = img_transformation
        self.target_transform = target_transformation
        self.data_root = data_root

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

        assert (
            labeled_loader.dataset.__len__() + unlabeled_loader.dataset.__len__()
            == train_set.__len__()
        )

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
            print(
                f">>>Generating {len(labeled_indices)} labeled data and {len(unlabeled_indices)} unlabeled data."
            )
        return labeled_indices, unlabeled_indices
