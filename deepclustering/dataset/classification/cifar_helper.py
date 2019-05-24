"""
This is a wrapper script to help to return the cifar dataloader.
"""
from functools import reduce
from typing import *

from .cifar import CIFAR10
from .clustering_helper import ClusterDatasetInterface
from ... import DATA_PATH
from ...augment import TransformInterface

__all__ = ['Cifar10DatasetInterface', 'default_cifar10_img_transform']


# todo try to extend the class to support semi supervised cases ....
class Cifar10DatasetInterface(ClusterDatasetInterface):
    """
    For unsupervised learning with parallel transformed datasets.
    """
    ALLOWED_SPLIT = ['train', 'val']

    def __init__(self, split_partitions: List[str] = ['train', 'val'], batch_size: int = 1, shuffle: bool = False,
                 num_workers: int = 1, pin_memory: bool = True) -> None:
        super().__init__(CIFAR10, split_partitions, batch_size, shuffle, num_workers, pin_memory)

    def _creat_concatDataset(self, image_transform: Callable, target_transform: Callable, dataset_dict: dict = {}):
        for split in self.split_partitions:
            assert split in self.ALLOWED_SPLIT, f"Allowed split in cifar-10:{self.ALLOWED_SPLIT}, given {split}."

        _datasets = []
        for split in self.split_partitions:
            dataset = self.DataClass(DATA_PATH, train=True if split == 'train' else False,
                                     transform=image_transform, target_transform=target_transform,
                                     download=True, **dataset_dict)
            _datasets.append(dataset)
        serial_dataset = reduce(lambda x, y: x + y, _datasets)
        return serial_dataset


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
    'tf1': {
        'randomcrop': {'size': (20, 20)},
        'Resize': {'size': (32, 32), 'interpolation': 0},
        'Img2Tensor': {'include_rgb': False, 'include_grey': True}
    },
    'tf2': {
        'randomcrop': {'size': (20, 20)},
        'Resize': {'size': (32, 32), 'interpolation': 0},
        'RandomHorizontalFlip': {'p': 0.5},
        'ColorJitter': {'brightness': [0.6, 1.4],
                        'contrast': [0.6, 1.4],
                        'saturation': [0.6, 1.4],
                        'hue': [-0.125, 0.125]},
        'Img2Tensor': {'include_rgb': False, 'include_grey': True}
    },
    'tf3': {
        'CenterCrop': {'size': (20, 20)},
        'Resize': {'size': (32, 32), 'interpolation': 0},
        'Img2Tensor': {'include_rgb': False, 'include_grey': True}
    }
}
default_cifar10_img_transform = {}
for k, v in transform_dict.items():
    default_cifar10_img_transform[k] = TransformInterface(v)

# parameters for the Cifar10ClusteringDataloaders. Used for clustering
# default_cifar10_img_transform = {
#     "tf1": transforms.Compose([
#         augment.RandomCrop(size=(20, 20)),
#         augment.Resize(size=(32, 32), interpolation=PIL.Image.BILINEAR),
#         augment.Img2Tensor(include_rgb=False, include_grey=True)
#     ]),
#     "tf2":
#         transforms.Compose([
#             augment.RandomCrop(size=(20, 20)),
#             augment.Resize(size=(32, 32), interpolation=PIL.Image.BILINEAR),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.ColorJitter(brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4],
#                                    hue=[-0.125, 0.125]),
#             augment.Img2Tensor(include_rgb=False, include_grey=True)
#         ]),
#     "tf3": transforms.Compose([
#         augment.CenterCrop(size=(20, 20)),
#         augment.Resize(size=(32, 32), interpolation=PIL.Image.BILINEAR),
#         augment.Img2Tensor(include_rgb=False, include_grey=True)
#     ])
# }

# todo: generate a custom function to generate the transform from yaml file.
# todo: add semi supervised interface
