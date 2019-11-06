# this is the toy example to optimize the primal-dual gradient descent.
# toy example for mnist dataset
import numpy as np
from torch.utils.data import Subset
from torchvision.datasets import MNIST

from deepclustering import DATA_PATH


def _override():
    pass


def _draw_equal_dataset(target: np.ndarray, num_samples: 1000) -> np.ndarray:
    """
    given the `target` and `num_samples`, return the labeled_index`
    :param target: target
    :param num_samples: 4000
    :return: labeled_index
    """
    total_classes: int = len(np.unique(target))
    num_per_class: int = int(num_samples / total_classes)
    labeled_index = []
    for _target in range(total_classes):
        labeled_index.extend(np.random.permutation(np.where(target == _target)[0])[:num_per_class].tolist())
    labeled_index.sort()
    assert len(labeled_index) == num_samples
    return np.array(labeled_index)


train_set = MNIST(root=DATA_PATH, train=True, download=True)
val_set = MNIST(root=DATA_PATH, train=False, download=True)
labeled_index = _draw_equal_dataset(train_set.targets, 1000)
labeled_set = Subset(train_set, labeled_index)
