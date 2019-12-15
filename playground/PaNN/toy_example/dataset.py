__all__ = ["get_mnist_dataloaders"]
from copy import deepcopy as dcp
from typing import *

import numpy as np
import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, Subset, DataLoader, RandomSampler
from torchvision import transforms
from torchvision.datasets import MNIST

from deepclustering import DATA_PATH


def _override_transformation(
    dataset: Union[Dataset, Subset], transform: Callable[[Image.Image], Tensor]
):
    """
    Iterative way to assign transform
    :param dataset:
    :param transform:
    :return:
    """
    assert isinstance(dataset, (MNIST, Subset))
    if isinstance(dataset, MNIST):
        dataset.transform = transform
    else:
        _override_transformation(dataset.dataset, transform)


def _draw_equal_dataset(
    target: np.ndarray, num_samples: int = 1000, allowed_classes: List[int] = None
) -> np.ndarray:
    """
    given the `target` and `num_samples`, return the labeled_index`
    :param target: target
    :param num_samples: 4000
    :param allowed_classes: None or list of targets like [0, 1, 2]
    :return: labeled_index
    """
    if allowed_classes is None:
        allowed_classes = list(range(len(np.unique(target))))
    total_classes = len(allowed_classes)
    num_per_class: int = int(num_samples / total_classes)
    labeled_index: List[int] = []
    for _target in allowed_classes:
        labeled_index.extend(
            np.random.permutation(np.where(target == _target)[0])[
                :num_per_class
            ].tolist()
        )
    labeled_index.sort()
    assert len(labeled_index) == num_samples
    return np.array(labeled_index)


def _draw_inequal_dataset(
    target: np.ndarray, class_sample_nums: Dict[int, int], excluded_index: List[int]
) -> np.ndarray:
    available_index = list(set(list(range(target.__len__()))) - set(excluded_index))
    return_list: List[int] = []
    for _target, sample_num in class_sample_nums.items():
        _target_index = np.where(target == _target)[0].tolist()
        _available_index = list(set(available_index) & set(_target_index))
        return_list.extend(
            np.random.permutation(_available_index)[:sample_num].tolist()
        )
    assert set(excluded_index) & set(return_list) == set()
    return np.array(return_list)


def show_dataset(dataset: Union[Subset, MNIST]):
    if isinstance(dataset, MNIST):
        print(dataset)
    else:
        print(dataset.dataset.__repr__())
        indice = dataset.indices
        try:
            targets = dataset.dataset.targets[indice]
        except:
            targets = dataset.dataset.targets[np.ndarray(indice)]
        print("label partition:")
        print(pd.Series(targets).value_counts())


def get_mnist_dataloaders(
    labeled_sample_num=10,
    unlabeled_class_sample_nums=None,
    train_transform=None,
    val_transform=None,
    dataloader_params={},
):

    train_set = MNIST(root=DATA_PATH, train=True, download=True)
    val_set = MNIST(root=DATA_PATH, train=False, download=True, transform=val_transform)
    val_set_index = _draw_equal_dataset(
        val_set.targets, num_samples=4000, allowed_classes=[0, 1, 2, 3, 4]
    )
    val_set = Subset(val_set, val_set_index)

    labeled_index = _draw_equal_dataset(
        train_set.targets, labeled_sample_num, allowed_classes=[0, 1, 2, 3, 4]
    )
    labeled_set = Subset(dcp(train_set), labeled_index)
    _override_transformation(labeled_set, train_transform)
    unlabeled_index = _draw_inequal_dataset(
        train_set.targets,
        class_sample_nums=unlabeled_class_sample_nums,
        excluded_index=labeled_index.tolist(),
    )
    unlabeled_set = Subset(dcp(train_set), unlabeled_index)
    _override_transformation(unlabeled_set, train_transform)
    assert set(labeled_index.tolist()) & set(unlabeled_index.tolist()) == set()
    del train_set

    show_dataset(labeled_set)
    show_dataset(unlabeled_set)
    show_dataset(val_set)

    labeled_loader = DataLoader(
        labeled_set,
        sampler=RandomSampler(
            data_source=labeled_set, replacement=True, num_samples=int(1e5)
        ),
        **dataloader_params
    )
    unlabeled_loader = DataLoader(
        unlabeled_set,
        sampler=RandomSampler(
            data_source=unlabeled_set, replacement=True, num_samples=int(1e5)
        ),
        **dataloader_params
    )
    val_loader = DataLoader(val_set, num_workers=1, batch_size=16)
    return labeled_loader, unlabeled_loader, val_loader
