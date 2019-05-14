from itertools import repeat
from typing import *

import PIL
from pathlib2 import Path
from torch.utils.data import DataLoader
from torchvision import transforms

from .mnist import MNIST
from .. import dataset  # type: ignore
from ...augment import augment

DATA_ROOT = str(Path(__file__).parents[3] / '.data')
Path(DATA_ROOT).mkdir(exist_ok=True)


class MNISTClusteringDataloaders(object):
    """
    dataset interface for unsupervised learning with combined train and test sets.
    return fixible dataloader with different transform functions, can be extended by creating subclasses for semi-supervised...
    """

    def __init__(self, batch_size: int = 1, shuffle: bool = False,
                 num_workers: int = 1, pin_memory: bool = False) -> None:
        """
        :param batch_size: batch_size = 1
        :param shuffle: shuffle the dataset, default = False
        :param num_workers: default 1
        """
        super().__init__()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    @staticmethod
    def _creat_concatDataset(image_transform: Callable, target_transform: Callable, dataset_dict: dict = {}):
        """
        create concat dataset with only one type of transform.
        :rtype: dataset
        :param image_transform:
        :param target_transform:
        :param dataset_dict:
        :return:
        """
        trainset = MNIST(root=DATA_ROOT, train=True, transform=image_transform, target_transform=target_transform,
                         download=True, **dataset_dict)
        valset = MNIST(root=DATA_ROOT, train=False, transform=image_transform, target_transform=target_transform,
                       download=True, **dataset_dict)

        concatSet = trainset + valset
        return concatSet

    '''
    def creat_ConcatDataLoader(self, image_transform: Callable = None, target_transform: Callable = None,
                               dataset_dict: Dict[str, Any] = {}, dataloader_dict: Dict[str, Any] = {}) -> DataLoader:
        r"""
        :param image_transform: Callable function for both tran and val
        :param target_transform: Callable function for target such as remapping
        :param dataset_dict: supplementary options for datasets
        :param dataloader_dict: supplementary options for dataloader
        :return: type: Dataloader
        """
        concatSet = self._creat_concatDataset(image_transform, target_transform, dataset_dict)
        concatLoader = DataLoader(concatSet, batch_size=self.batch_size, shuffle=self.shuffle,
                                  num_workers=self.num_workers, drop_last=True, pin_memory=self.pin_memory,
                                  **dataloader_dict)
        return concatLoader
    '''

    def _creat_combineDataset(self, image_transforms: Tuple[Callable, ...], target_transform: Callable = None,
                              dataset_dict: Dict[str, Any] = {}):
        assert len(image_transforms) >= 1, f"Given {image_transforms}"
        assert not isinstance(target_transform,
                              (list, tuple)), f"We consider the target_transform should be the same for all."
        concatSets = []
        for t_img, t_tar in zip(image_transforms, repeat(target_transform)):
            concatSets.append(
                self._creat_concatDataset(image_transform=t_img, target_transform=t_tar, dataset_dict=dataset_dict))
        combineSet = dataset.CombineDataset(*concatSets)
        return combineSet

    def creat_CombineDataLoader(self, *image_transforms: Callable, target_transform: Callable = None,
                                dataset_dict: Dict[str, Any] = {}, dataloader_dict: Dict[str, Any] = {}) -> DataLoader:
        combineSet = self._creat_combineDataset(image_transforms, target_transform, dataset_dict)
        combineLoader = DataLoader(combineSet, batch_size=self.batch_size, shuffle=self.shuffle,
                                   num_workers=self.num_workers, drop_last=True, pin_memory=self.pin_memory,
                                   **dataloader_dict)
        return combineLoader


## default transform
default_mnist_img_transform = {
    "tf1": transforms.Compose([
        augment.CenterCrop(size=(26, 26)),
        augment.Resize(size=(28, 28), interpolation=PIL.Image.BILINEAR),
        augment.Img2Tensor(include_rgb=False, include_grey=True),
        transforms.Normalize((0.5,), (0.5,))
    ]),
    "tf2":
        transforms.Compose([
            augment.RandomCrop(size=(26, 26), padding=2),
            augment.Resize(size=(28, 28), interpolation=PIL.Image.BILINEAR),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4],
                                   hue=[-0.125, 0.125]),
            augment.Img2Tensor(include_rgb=False, include_grey=True),
            transforms.Normalize((0.5,), (0.5,))
        ]),
    "tf3": transforms.Compose([
        augment.CenterCrop(size=(26, 26)),
        augment.Resize(size=(28, 28), interpolation=PIL.Image.BILINEAR),
        augment.Img2Tensor(include_rgb=False, include_grey=True),
        transforms.Normalize((0.5,), (0.5,))
    ])
}
