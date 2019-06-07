__all__ = ['MNISTDatasetInterface', 'default_mnist_img_transform']

from functools import reduce
from typing import *

import PIL
from torchvision import transforms

from .clustering_helper import ClusterDatasetInterface
from .mnist import MNIST
from ... import DATA_PATH
from ...augment import augment


class MNISTDatasetInterface(ClusterDatasetInterface):
    """
    dataset interface for unsupervised learning with combined train and test sets.
    """
    ALLOWED_SPLIT = ['train', 'val']

    def __init__(self, split_partitions: List[str] = ['train', 'val'], batch_size: int = 1, shuffle: bool = False,
                 num_workers: int = 1, pin_memory: bool = True, drop_last=False) -> None:
        super().__init__(MNIST, split_partitions, batch_size, shuffle, num_workers, pin_memory, drop_last)

    def _creat_concatDataset(self, image_transform: Callable, target_transform: Callable, dataset_dict: dict = {}):
        for split in self.split_partitions:
            assert split in self.ALLOWED_SPLIT, f"Allowed split in MNIST-10:{self.ALLOWED_SPLIT}, given {split}."

        _datasets = []
        for split in self.split_partitions:
            dataset = self.DataClass(DATA_PATH, train=True if split == 'train' else False,
                                     transform=image_transform, target_transform=target_transform,
                                     download=True, **dataset_dict)
            _datasets.append(dataset)
        serial_dataset = reduce(lambda x, y: x + y, _datasets)
        return serial_dataset


default_mnist_img_transform = {
    "tf1":
        transforms.Compose([
            augment.RandomChoice(transforms=
                                 [augment.RandomCrop(size=(20, 20), padding=None),
                                  augment.CenterCrop(size=(20, 20))
                                  ]),
            augment.Resize(size=24, interpolation=PIL.Image.BILINEAR),
            transforms.ToTensor()
        ]),
    "tf2":
        transforms.Compose([
            augment.RandomApply(transforms=[transforms.RandomRotation(
                degrees=(-25.0, 25.0),
                resample=False,
                expand=False
            )], p=0.5),
            augment.RandomChoice(transforms=[
                augment.RandomCrop(size=(16, 16), padding=None),
                augment.RandomCrop(size=(20, 20), padding=None),
                augment.RandomCrop(size=(24, 24), padding=None),
            ]),
            augment.Resize(size=24, interpolation=PIL.Image.BILINEAR),
            transforms.ColorJitter(brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4],
                                   hue=[-0.125, 0.125]),
            transforms.ToTensor()
        ]),
    "tf3":
        transforms.Compose([
            augment.CenterCrop(size=(20, 20)),
            augment.Resize(size=24, interpolation=PIL.Image.BILINEAR),
            transforms.ToTensor()
        ]),

}

""" Taken from the IIC paper
tf1
Compose(
    RandomChoice(
    RandomCrop(size=(20, 20), padding=None)
    CenterCrop(size=(20, 20))
)
    Resize(size=24, interpolation=PIL.Image.BILINEAR)
    ToTensor()
)
tf2
Compose(
    RandomApply(
        p=0.5
        RandomRotation(degrees=(-25.0, 25.0), resample=False, expand=False)
        )
    RandomChoice(
        RandomCrop(size=(16, 16), padding=None)
        RandomCrop(size=(20, 20), padding=None)
        RandomCrop(size=(24, 24), padding=None)
    )
    Resize(size=(24, 24), interpolation=PIL.Image.BILINEAR)
    ColorJitter(
        brightness=[0.6, 1.4], 
        contrast=[0.6, 1.4], 
        saturation=[0.6, 1.4], 
        hue=[-0.125, 0.125]
    )
    ToTensor()
)
tf3:
Compose(
    CenterCrop(size=(20, 20))
    Resize(size=24, interpolation=PIL.Image.BILINEAR)
    ToTensor()
)
"""

# default transform
# default_mnist_img_transform = {
#     "tf1": transforms.Compose([
#         augment.CenterCrop(size=(26, 26)),
#         augment.Resize(size=(28, 28), interpolation=PIL.Image.BILINEAR),
#         augment.Img2Tensor(include_rgb=False, include_grey=True),
#         transforms.Normalize((0.5,), (0.5,))
#     ]),
#     "tf2":
#         transforms.Compose([
#             augment.RandomCrop(size=(26, 26), padding=2),
#             augment.Resize(size=(28, 28), interpolation=PIL.Image.BILINEAR),
#             # transforms.RandomHorizontalFlip(p=0.5),
#             transforms.ColorJitter(brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4],
#                                    hue=[-0.125, 0.125]),
#             augment.Img2Tensor(include_rgb=False, include_grey=True),
#             transforms.Normalize((0.5,), (0.5,))
#         ]),
#     "tf3": transforms.Compose([
#         augment.CenterCrop(size=(26, 26)),
#         augment.Resize(size=(28, 28), interpolation=PIL.Image.BILINEAR),
#         augment.Img2Tensor(include_rgb=False, include_grey=True),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
# }
