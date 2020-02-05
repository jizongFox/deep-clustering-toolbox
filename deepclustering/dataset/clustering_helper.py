from abc import abstractmethod
from itertools import repeat
from typing import *

from torch.utils.data import Dataset, DataLoader

from deepclustering.dataloader import dataset


class ClusterDatasetInterface(object):
    """
    dataset interface for unsupervised learning with combined train and test sets.
    """

    ALLOWED_SPLIT = []

    def __init__(
        self,
        DataClass: Dataset,
        data_root: str,
        split_partitions: List[str],
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 1,
        pin_memory: bool = True,
        drop_last=False,
    ) -> None:
        """
        :param batch_size: batch_size = 1
        :param shuffle: shuffle the dataset, default = False
        :param num_workers: default 1
        """
        super().__init__()
        self.DataClass = DataClass
        if not isinstance(split_partitions, list):
            split_partitions = [split_partitions]
        assert isinstance(split_partitions[0], str), (
            f"Elements of split_partitions must be str, " f"given {split_partitions[0]}"
        )
        self.split_partitions = split_partitions
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.data_root = data_root

    @abstractmethod
    def _creat_concatDataset(
        self,
        image_transform: Callable,
        target_transform: Optional[Callable],
        dataset_dict: dict = {},
    ):
        """
        create concat dataset with only one type of transform.
        :rtype: dataset
        :param image_transform:
        :param target_transform:
        :param dataset_dict:
        :return:
        """
        raise NotImplementedError

    def _creat_combineDataset(
        self,
        image_transforms: Tuple[Callable, ...],
        target_transform: Tuple[Callable, ...] = None,
        dataset_dict: Dict[str, Any] = {},
    ):
        if target_transform is None:
            assert len(image_transforms) >= 1, f"Given {image_transforms}"
            target_transform = repeat(target_transform)
        elif len(target_transform) == 1 and not isinstance(target_transform, list):
            target_transform = repeat(target_transform)
        else:
            assert len(image_transforms) == len(target_transform)
        concatSets = []
        for t_img, t_tar in zip(image_transforms, target_transform):
            concatSets.append(
                self._creat_concatDataset(
                    image_transform=t_img,
                    target_transform=t_tar,
                    dataset_dict=dataset_dict,
                )
            )
        combineSet = dataset.CombineDataset(*concatSets)
        return combineSet

    def SerialDataLoader(
        self,
        image_transform: Callable = None,
        target_transform: Callable = None,
        dataset_dict: Dict[str, Any] = {},
        dataloader_dict: Dict[str, Any] = {},
    ) -> DataLoader:
        r"""
        Combine several dataset in a serial way.
        :param image_transform: Callable function for both tran and val
        :param target_transform: Callable function for target such as remapping
        :param dataset_dict: supplementary options for datasets
        :param dataloader_dict: supplementary options for dataloader
        :return: type: Dataloader
        """
        concatSet = self._creat_concatDataset(
            image_transform, target_transform, dataset_dict
        )
        concatLoader = DataLoader(
            concatSet,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            **dataloader_dict,
        )
        return concatLoader

    def ParallelDataLoader(
        self,
        *image_transforms: Callable,
        target_transform: Union[Callable, Tuple[Callable, ...]] = None,
        dataset_dict: Dict[str, Any] = {},
        dataloader_dict: Dict[str, Any] = {},
    ) -> DataLoader:
        parallel_set = self._creat_combineDataset(
            image_transforms, target_transform, dataset_dict
        )
        parallel_loader = DataLoader(
            parallel_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            **dataloader_dict,
        )
        return parallel_loader
