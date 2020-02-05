"""
This is a wrapper script for semi supervised leanring and clustering for svhn dataset.
"""
__all__ = [
    "SVHNClusteringDatasetInterface",
    "SVHNSemiSupervisedDatasetInterface",
    "svhn_naive_transform",
    "svhn_strong_transform",
]

from functools import reduce
from typing import List, Callable, Tuple

import PIL
from torch.utils.data import Dataset
from torchvision import transforms

from deepclustering.augment import pil_augment
from .svhn import SVHN
from ..clustering_helper import ClusterDatasetInterface
from ..semi_helper import SemiDataSetInterface
from ... import DATA_PATH


class SVHNSemiSupervisedDatasetInterface(SemiDataSetInterface):
    def __init__(
        self,
        data_root: str = DATA_PATH,
        labeled_sample_num: int = 1000,
        seed: int = 0,
        batch_size: int = 10,
        labeled_batch_size: int = None,
        unlabeled_batch_size: int = None,
        val_batch_size: int = None,
        shuffle: bool = False,
        num_workers: int = 1,
        pin_memory: bool = True,
        drop_last=False,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            SVHN,
            data_root,
            labeled_sample_num,
            seed,
            batch_size,
            labeled_batch_size,
            unlabeled_batch_size,
            val_batch_size,
            shuffle,
            num_workers,
            pin_memory,
            drop_last,
            verbose,
        )

    def _init_train_val_sets(self) -> Tuple[Dataset, Dataset]:
        train_set = self.DataClass(self.data_root, split="train", download=True)
        val_set = self.DataClass(self.data_root, split="test", download=True)
        return train_set, val_set


class SVHNClusteringDatasetInterface(ClusterDatasetInterface):
    ALLOWED_SPLIT = ["train", "test"]

    def __init__(
        self,
        data_root=None,
        split_partitions: List[str] = [],
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 1,
        pin_memory: bool = True,
    ) -> None:
        super().__init__(
            SVHN,
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
            ), f"Allowed split in SVHN:{self.ALLOWED_SPLIT}, given {split}."

        _datasets = []
        for split in self.split_partitions:
            dataset = self.DataClass(
                self.data_root,
                split=split,
                transform=image_transform,
                target_transform=target_transform,
                download=True,
                **dataset_dict,
            )
            _datasets.append(dataset)
        serial_dataset = reduce(lambda x, y: x + y, _datasets)
        return serial_dataset


# ===================== public transform interface ===========================
svhn_naive_transform = {
    # output size 32*32
    "tf1": transforms.Compose([pil_augment.Img2Tensor()]),
    "tf2": transforms.Compose(
        [pil_augment.RandomCrop(size=32, padding=2), pil_augment.Img2Tensor()]
    ),
    "tf3": transforms.Compose([pil_augment.Img2Tensor()]),
}
svhn_strong_transform = {
    # output size 32*32
    "tf1": transforms.Compose(
        [
            pil_augment.CenterCrop(size=(28, 28)),
            pil_augment.Resize(size=32, interpolation=PIL.Image.BILINEAR),
            pil_augment.Img2Tensor(),
        ]
    ),
    "tf2": transforms.Compose(
        [
            pil_augment.RandomApply(
                transforms=[
                    transforms.RandomRotation(
                        degrees=(-25.0, 25.0), resample=False, expand=False
                    )
                ],
                p=0.5,
            ),
            pil_augment.RandomChoice(
                transforms=[
                    pil_augment.RandomCrop(size=(20, 20), padding=None),
                    pil_augment.RandomCrop(size=(24, 24), padding=None),
                    pil_augment.RandomCrop(size=(28, 28), padding=None),
                ]
            ),
            pil_augment.Resize(size=32, interpolation=PIL.Image.BILINEAR),
            transforms.ColorJitter(
                brightness=[0.6, 1.4],
                contrast=[0.6, 1.4],
                saturation=[0.6, 1.4],
                hue=[-0.125, 0.125],
            ),
            pil_augment.Img2Tensor(),
        ]
    ),
    "tf3": transforms.Compose(
        [
            pil_augment.CenterCrop(size=(28, 28)),
            pil_augment.Resize(size=32, interpolation=PIL.Image.BILINEAR),
            pil_augment.Img2Tensor(),
        ]
    ),
}
# ============================================================================================
