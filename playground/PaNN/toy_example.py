# this is the toy example to optimize the primal-dual gradient descent.
# toy example for mnist dataset
from copy import deepcopy as dcp
from typing import Dict, List, Union, Callable

import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Subset, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

from deepclustering import DATA_PATH


def _override_transformation(dataset: Union[Dataset, Subset], transform: Callable[[Image.Image], Tensor]):
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


def _draw_equal_dataset(target: np.ndarray, num_samples: int = 1000, allowed_classes: List[int] = None) -> np.ndarray:
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
        labeled_index.extend(np.random.permutation(np.where(target == _target)[0])[:num_per_class].tolist())
    labeled_index.sort()
    assert len(labeled_index) == num_samples
    return np.array(labeled_index)


def _draw_inequal_dataset(
        target: np.ndarray,
        class_sample_nums: Dict[int, int],
        excluded_index: List[int]
) -> np.ndarray:
    available_index = list(set(list(range(target.__len__()))) - set(excluded_index))
    return_list: List[int] = []
    for _target, sample_num in class_sample_nums.items():
        _target_index = np.where(target == _target)[0].tolist()
        _available_index = list(set(available_index) - set(_target_index))
        return_list.extend(np.random.permutation(_available_index)[:sample_num].tolist())
    assert set(excluded_index) & set(return_list) == set()
    return np.array(return_list)


def show_dataset(dataset:Subset):
    print(dataset.dataset)


unlabeled_class_sample_nums = {
    1: 1000,
    2: 2000,
    3: 3000,
    4: 4000
}
train_transform = transforms.Compose([
    transforms.RandomCrop((28, 28), padding=2),
    transforms.ToTensor()
])
val_transform = transforms.Compose([
    transforms.ToTensor()
])

train_set = MNIST(root=DATA_PATH, train=True, download=True)
val_set = MNIST(root=DATA_PATH, train=False, download=True)
labeled_index = _draw_equal_dataset(train_set.targets, 1000, allowed_classes=[1, 2, 3, 4])
labeled_set = Subset(dcp(train_set), labeled_index)

unlabeled_index = _draw_inequal_dataset(train_set.targets, class_sample_nums=unlabeled_class_sample_nums,
                                        excluded_index=labeled_index.tolist())
unlabeled_set = Subset(dcp(train_set), unlabeled_index)
assert set(labeled_index.tolist()) & set(unlabeled_index.tolist()) == set()
show_dataset(labeled_set)
show_dataset(unlabeled_set)
