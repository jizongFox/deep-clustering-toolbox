"""
This is a wrapper script to help to return the cifar dataloader.
"""

__all__ = [
    "Cifar10ClusteringDatasetInterface",
    "Cifar10SemiSupervisedDatasetInterface",
    "default_cifar10_img_transform",
]
from functools import reduce
from typing import *

from deepclustering.dataset.clustering_helper import ClusterDatasetInterface
from deepclustering.dataset.semi_helper import SemiDatasetInterface, SemiDataSetInterface_
from torch.utils.data import Dataset

from .cifar import CIFAR10
from ... import DATA_PATH
from ...augment import TransformInterface


class Cifar10ClusteringDatasetInterface(ClusterDatasetInterface):
    """
    For unsupervised learning with parallel transformed datasets.
    """

    ALLOWED_SPLIT = ["train", "val"]

    def __init__(
            self,
            data_root=DATA_PATH,
            split_partitions: List[str] = ["train", "val"],
            batch_size: int = 1,
            shuffle: bool = False,
            num_workers: int = 1,
            pin_memory: bool = True,
    ) -> None:
        super().__init__(
            CIFAR10,
            data_root,
            split_partitions,
            batch_size,
            shuffle,
            num_workers,
            pin_memory,
        )

    def _creat_concatDataset(
            self,
            image_transform: Callable,
            target_transform: Callable,
            dataset_dict: dict = {},
    ):
        for split in self.split_partitions:
            assert (
                    split in self.ALLOWED_SPLIT
            ), f"Allowed split in cifar-10:{self.ALLOWED_SPLIT}, given {split}."

        _datasets = []
        for split in self.split_partitions:
            dataset = self.DataClass(
                DATA_PATH,
                train=True if split == "train" else False,
                transform=image_transform,
                target_transform=target_transform,
                download=True,
                **dataset_dict,
            )
            _datasets.append(dataset)
        serial_dataset = reduce(lambda x, y: x + y, _datasets)
        return serial_dataset


class Cifar10SemiSupervisedDatasetInterface(SemiDatasetInterface):
    def __init__(
            self,
            data_root=DATA_PATH,
            labeled_sample_num: int = 4000,
            img_transformation: Callable = None,
            target_transformation: Callable = None,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(
            CIFAR10,
            data_root,
            labeled_sample_num,
            img_transformation,
            target_transformation,
            *args,
            **kwargs,
        )

    def _init_train_and_test_test(
            self, transform, target_transform, *args, **kwargs
    ) -> Tuple[Dataset, Dataset]:
        super()._init_train_and_test_test(transform, target_transform, *args, **kwargs)
        train_set = self.DataClass(
            DATA_PATH,
            train=True,
            transform=transform,
            target_transform=target_transform,
            download=True,
            *args,
            **kwargs,
        )
        val_set = self.DataClass(
            DATA_PATH,
            train=False,
            transform=transform,
            target_transform=target_transform,
            download=True,
            *args,
            **kwargs,
        )
        return train_set, val_set


class Cifar10SemiSupervisedParallelDatasetInterface(SemiDataSetInterface_):

    def __init__(self, data_root: str, labeled_sample_num: int = 4000, seed: int = 0, batch_size: int = 1,
                 shuffle: bool = False, num_workers: int = 1, pin_memory: bool = True, drop_last=False) -> None:
        super().__init__(CIFAR10, data_root, labeled_sample_num, seed, batch_size, shuffle, num_workers, pin_memory,
                         drop_last)

    def _init_train_val_sets(self) -> Tuple[Dataset, Dataset]:
        train_set = self.DataClass(
            DATA_PATH,
            train=True,
            download=True,
        )
        val_set = self.DataClass(
            DATA_PATH,
            train=False,
            download=True,
        )
        return train_set, val_set


# taken from IIC paper:
r"""
tf1=Compose(a
        RandomCrop(size=(20, 20), padding=None)
        Resize(size=(32, 32), interpolation=PIL.Image.BILINEAR)
        <function custom_greyscale_to_tensor.<locals>._inner at 0x7f2d1d099d90>
    )
tf2=Compose(
        RandomCrop(size=(20, 20), padding=None)
        Resize(size=(32, 32), interpolation=PIL.Image.BILINEAR)
        RandomHorizontalFlip(p=0.5)
        ColorJitter(brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4], hue=[-0.125, 0.125])
        <function custom_greyscale_to_tensor.<locals>._inner at 0x7f2c8cc57f28>
    )
tf3=Compose(
        CenterCrop(size=(20, 20))
        Resize(size=(32, 32), interpolation=PIL.Image.BILINEAR)
        <function custom_greyscale_to_tensor.<locals>._inner at 0x7f2c8cc57ea0>
    )
"""
# convert to dictionary configuration:
transform_dict = {
    "tf1": {
        "randomcrop": {"size": (20, 20)},
        "Resize": {"size": (32, 32), "interpolation": 0},
        "Img2Tensor": {"include_rgb": False, "include_grey": True},
    },
    "tf2": {
        "randomcrop": {"size": (20, 20)},
        "Resize": {"size": (32, 32), "interpolation": 0},
        "RandomHorizontalFlip": {"p": 0.5},
        "ColorJitter": {
            "brightness": [0.6, 1.4],
            "contrast": [0.6, 1.4],
            "saturation": [0.6, 1.4],
            "hue": [-0.125, 0.125],
        },
        "Img2Tensor": {"include_rgb": False, "include_grey": True},
    },
    "tf3": {
        "CenterCrop": {"size": (20, 20)},
        "Resize": {"size": (32, 32), "interpolation": 0},
        "Img2Tensor": {"include_rgb": False, "include_grey": True},
    },
}
default_cifar10_img_transform = {}
for k, v in transform_dict.items():
    default_cifar10_img_transform[k] = TransformInterface(v)
