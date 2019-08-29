from typing import Callable

from .toydataset import Cls_ShapesDataset
from deepclustering.dataset.clustering_helper import ClusterDatasetInterface


class ToyExampleInterFace(ClusterDatasetInterface):
    ALLOWED_SPLIT = ["1"]

    def __init__(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 1,
        pin_memory: bool = True,
        drop_last=False,
    ) -> None:
        super().__init__(
            Cls_ShapesDataset,
            "",
            ["1"],
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
        train_set = Cls_ShapesDataset(
            count=5000,
            height=100,
            width=100,
            max_object_scale=0.75,
            transform=image_transform,
            target_transform=target_transform,
            **dataset_dict
        )
        return train_set
