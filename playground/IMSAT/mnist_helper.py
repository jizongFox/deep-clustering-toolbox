__all__ = [
    "MNISTClusteringDatasetInterface",
    "MNISTSemiSupervisedDatasetInterface",
    "default_mnist_img_transform",
]
from functools import reduce
from typing import *

import PIL
from torch.utils.data import Dataset
from torchvision import transforms

from deepclustering.dataset.clustering_helper import ClusterDatasetInterface
from deepclustering.dataset.semi_helper import SemiDatasetInterface
from .mnist import MNIST
from deepclustering import DATA_PATH
from deepclustering.augment import pil_augment


class MNISTClusteringDatasetInterface(ClusterDatasetInterface):
    """
    dataset interface for unsupervised learning with combined train and test sets.
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
        drop_last=False,
    ) -> None:
        super().__init__(
            MNIST,
            data_root,
            split_partitions,
            batch_size,
            shuffle,
            num_workers,
            pin_memory,
            drop_last,
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
            ), f"Allowed split in MNIST-10:{self.ALLOWED_SPLIT}, given {split}."

        _datasets = []
        for split in self.split_partitions:
            dataset = self.DataClass(
                self.data_root,
                train=True if split == "train" else False,
                transform=image_transform,
                target_transform=target_transform,
                download=True,
                **dataset_dict,
            )
            _datasets.append(dataset)
        serial_dataset = reduce(lambda x, y: x + y, _datasets)
        return serial_dataset


class MNISTSemiSupervisedDatasetInterface(SemiDatasetInterface):
    def __init__(
        self,
        data_root: str = DATA_PATH,
        labeled_sample_num: int = 100,
        img_transformation: Callable = None,
        target_transformation: Callable = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            MNIST,
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
        train_set = self.DataClass(
            self.data_root,
            train=True,
            transform=self.img_transform,
            target_transform=self.target_transform,
            download=True,
            *args,
            **kwargs,
        )
        val_set = self.DataClass(
            self.data_root,
            train=False,
            transform=self.img_transform,
            target_transform=self.target_transform,
            download=True,
            *args,
            **kwargs,
        )
        return train_set, val_set


default_mnist_img_transform = {
    "tf1": transforms.Compose(
        [
            pil_augment.RandomChoice(
                transforms=[
                    pil_augment.RandomCrop(size=(20, 20), padding=None),
                    pil_augment.CenterCrop(size=(20, 20)),
                ]
            ),
            pil_augment.Resize(size=24, interpolation=PIL.Image.BILINEAR),
            transforms.ToTensor(),
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
                    pil_augment.RandomCrop(size=(16, 16), padding=None),
                    pil_augment.RandomCrop(size=(20, 20), padding=None),
                    pil_augment.RandomCrop(size=(24, 24), padding=None),
                ]
            ),
            pil_augment.Resize(size=24, interpolation=PIL.Image.BILINEAR),
            transforms.ColorJitter(
                brightness=[0.6, 1.4],
                contrast=[0.6, 1.4],
                saturation=[0.6, 1.4],
                hue=[-0.125, 0.125],
            ),
            transforms.ToTensor(),
        ]
    ),
    "tf3": transforms.Compose(
        [
            pil_augment.CenterCrop(size=(20, 20)),
            pil_augment.Resize(size=24, interpolation=PIL.Image.BILINEAR),
            transforms.ToTensor(),
        ]
    ),
}
