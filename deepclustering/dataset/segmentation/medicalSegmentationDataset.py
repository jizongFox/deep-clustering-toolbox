from __future__ import print_function, division

import os
from functools import reduce
from operator import and_
from pathlib import Path
from typing import Callable, List, Tuple

from PIL import Image
from deepclustering import ModelMode
from deepclustering.augment import SequentialWrapper
from deepclustering.augment.pil_augment import ToTensor, ToLabel
from deepclustering.utils import map_
from torch import Tensor
from torch.utils.data import Dataset


def allow_extension(path: str, extensions: List[str]) -> bool:
    try:
        return Path(path).suffixes[0] in extensions
    except:
        return False


class MedicalImageSegmentationDataset(Dataset):
    dataset_modes = ["train", "val", "test", "unlabeled"]
    allow_extension = [".jpg", ".png"]

    def __init__(
            self,
            root_dir: str,
            mode: str,
            subfolders: List[str],
            transforms=None,
            verbose=True,
    ) -> None:
        assert (
                len(subfolders) == set(subfolders).__len__()
        ), f"subfolders must be unique, given {subfolders}."
        assert reduce(
            and_, [isinstance(s, str) for s in subfolders]
        ), f"subfolder elements should be str, given {subfolders}"

        subfolders = [subfolders] if isinstance(subfolders, str) else subfolders
        self.name: str = f"{mode}_dataset"
        self.mode: str = mode
        self.root_dir = root_dir
        self.subfolders: List[str] = subfolders
        self.transform: SequentialWrapper = transforms if transforms else SequentialWrapper(
            img_transform=ToTensor(),
            target_transform=ToLabel(),
            if_is_target=[False] + [True for _ in range(len(subfolders) - 1)],
        )
        self.verbose = verbose
        if verbose:
            print(f"->> Building {self.name}:\t")
        self.imgs, self.filenames = self.make_dataset(
            self.root_dir, self.mode, self.subfolders, verbose=verbose
        )

    def __len__(self) -> int:
        return int(len(self.imgs[self.subfolders[0]]))

    def set_mode(self, mode) -> None:
        assert isinstance(
            mode, (str, ModelMode)
        ), "the type of mode should be str or ModelMode, given %s" % str(mode)

        if isinstance(mode, str):
            self.training = ModelMode.from_str(mode)
        else:
            self.training = mode

    def __getitem__(self, index) -> Tuple[List[Tensor], str]:
        img_list, filename_list = self._getitem_index(index)
        assert img_list.__len__() == self.subfolders.__len__()
        # make sure the filename is the same image
        assert (
                set(map_(lambda x: Path(x).stem, filename_list)).__len__() == 1
        ), f"Check the filename list, given {filename_list}."
        filename = Path(filename_list[0]).stem
        img_list = self.transform(*img_list)
        return img_list, filename

    def _getitem_index(self, index):
        img_list = [
            Image.open(self.imgs[subfolder][index]) for subfolder in self.subfolders
        ]
        filename_list = [
            self.filenames[subfolder][index] for subfolder in self.subfolders
        ]
        return img_list, filename_list

    @classmethod
    def make_dataset(cls, root: str, mode: str, subfolders: List[str], verbose=True):
        assert mode in cls.dataset_modes
        for subfolder in subfolders:
            assert Path(root, mode, subfolder).exists(), os.path.join(
                root, mode, subfolder
            )
        items = [
            os.listdir(Path(os.path.join(root, mode, subfoloder)))
            for subfoloder in subfolders
        ]
        # clear up extension
        items = sorted(
            [
                [x for x in item if allow_extension(x, cls.allow_extension)]
                for item in items
            ]
        )
        assert set(map_(len, items)).__len__() == 1, map_(len, items)
        imgs = {}

        for subfolder, item in zip(subfolders, items):
            imgs[subfolder] = sorted(
                [os.path.join(root, mode, subfolder, x_path) for x_path in item]
            )
        assert set(map_(len, imgs.values())).__len__() == 1
        for subfolder in subfolders:
            if verbose:
                print(f"found {len(imgs[subfolder])} images in {subfolder}\t")
        return imgs, imgs


class MedicalImageSegmentationDatasetWithMetaInfo(MedicalImageSegmentationDataset):
    def __init__(
            self,
            root_dir: str,
            mode: str,
            subfolders: List[str],
            transforms=None,
            verbose=True,
            metainfo_generator: Callable = None,
    ) -> None:
        super().__init__(root_dir, mode, subfolders, transforms, verbose)
        self.metainfo_generator = metainfo_generator

    def __getitem__(self, index) -> Tuple[List[Tensor], str]:
        img_list, filename_list = (
            [Image.open(self.imgs[subfolder][index]) for subfolder in self.subfolders],
            [self.filenames[subfolder][index] for subfolder in self.subfolders],
        )
        assert img_list.__len__() == self.subfolders.__len__()
        # make sure the filename is the same image
        assert (
                set(map_(lambda x: Path(x).stem, filename_list)).__len__() == 1
        ), f"Check the filename list, given {filename_list}."
        filename = Path(filename_list[0]).stem
        img_list = self.transform(img_list)
        if self.metainfo_generator:
            metainfo = self.metainfo_generator(img_list)

        return img_list, filename, metainfo
