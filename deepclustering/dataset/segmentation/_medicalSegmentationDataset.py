import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from deepclustering.augment import SequentialWrapper
from deepclustering.augment.pil_augment import ToTensor, ToLabel
from deepclustering.utils import map_, assert_list


def allow_extension(path: str, extensions: List[str]) -> bool:
    try:
        return Path(path).suffixes[0] in extensions
    except:
        return False


def default_transform(subfolders) -> SequentialWrapper:
    return SequentialWrapper(
        img_transform=ToTensor(),
        target_transform=ToLabel(),
        if_is_target=[False] + [True] * (len(subfolders) - 1),
    )


class MedicalImageSegmentationDataset(Dataset):
    dataset_modes = ["train", "val", "test", "unlabeled"]
    allow_extension = [".jpg", ".png"]

    def __init__(
        self,
        root_dir: str,
        mode: str,
        subfolders: List[str],
        transforms: SequentialWrapper = None,
        patient_pattern: str = None,
        verbose=True,
    ) -> None:
        """
        :param root_dir: main folder path of the dataset
        :param mode: the subfolder name of this root, usually train, val, test or etc.
        :param subfolders: subsubfolder name of this root, usually img, gt, etc
        :param transforms: synchronized transformation for all the subfolders
        :param verbose: verbose
        """
        assert (
            len(subfolders) == set(subfolders).__len__()
        ), f"subfolders must be unique, given {subfolders}."
        assert assert_list(
            lambda x: isinstance(x, str), subfolders
        ), f"`subfolder` elements should be str, given {subfolders}"
        self._name: str = f"{mode}_dataset"
        self._mode: str = mode
        self._root_dir = root_dir
        self._subfolders: List[str] = subfolders
        self._transform = default_transform(self._subfolders)
        if transforms:
            self._transform = transforms
        self._verbose = verbose
        if self._verbose:
            print(f"->> Building {self._name}:\t")
        self._filenames = self._make_dataset(
            self._root_dir, self._mode, self._subfolders, verbose=verbose
        )
        self._debug = os.environ.get("PYDEBUG", "0") == "1"
        self._set_patient_pattern(patient_pattern)

    @property
    def subfolders(self) -> List[str]:
        return self._subfolders

    @property
    def verbose(self) -> bool:
        return self._verbose

    @property
    def is_debug(self) -> bool:
        return self._debug

    def get_filenames(self, subfolder_name=None) -> List[str]:
        if subfolder_name:
            return self._filenames[subfolder_name]
        else:
            return self._filenames[self.subfolders[0]]

    @property
    def dataset_pattern(self):
        return self._pattern

    @property
    def mode(self) -> str:
        return self._mode

    def __len__(self) -> int:
        if self._debug:
            return int(len(self._filenames[self.subfolders[0]]) / 10)
        return int(len(self._filenames[self.subfolders[0]]))

    def __getitem__(self, index) -> Tuple[List[Tensor], str]:
        img_list, filename_list = self._getitem_index(index)
        assert img_list.__len__() == self.subfolders.__len__()
        # make sure the filename is the same image
        assert (
            set(map_(lambda x: Path(x).stem, filename_list)).__len__() == 1
        ), f"Check the filename list, given {filename_list}."
        filename = Path(filename_list[0]).stem
        img_list = self._transform(*img_list)
        return img_list, filename

    def _getitem_index(self, index):
        img_list = [
            Image.open(self._filenames[subfolder][index])
            for subfolder in self.subfolders
        ]
        filename_list = [
            self._filenames[subfolder][index] for subfolder in self.subfolders
        ]
        return img_list, filename_list

    def _set_patient_pattern(self, pattern):
        """
        This set patient_pattern using re library.
        :param pattern:
        :return:
        """
        assert isinstance(pattern, str), pattern
        self._pattern = pattern
        self._re_pattern = re.compile(self._pattern)

    def _get_group_name(self, path: Union[Path, str]) -> str:
        if not hasattr(self, "_re_pattern"):
            raise RuntimeError(
                "Calling `_get_group_name` before setting `set_patient_pattern`"
            )
        if isinstance(path, str):
            path = Path(path)
        try:
            group_name = self._re_pattern.search(path.stem).group(0)
        except AttributeError:
            raise AttributeError(
                f"Cannot match pattern: {self._pattern} for path: {str(path)}"
            )
        return group_name

    def get_group_list(self):

        return sorted(
            list(
                set(
                    [
                        self._get_group_name(filename)
                        for filename in self.get_filenames()
                    ]
                )
            )
        )

    def set_transform(self, transform: SequentialWrapper) -> None:
        if not isinstance(transform, SequentialWrapper):
            raise TypeError(
                f"`transform` must be instance of `SequentialWrapper`, given {type(transform)}."
            )
        self._transform = transform

    @property
    def transform(self) -> Optional[SequentialWrapper]:
        return self._transform

    @classmethod
    def _make_dataset(
        cls, root: str, mode: str, subfolders: List[str], verbose=True
    ) -> Dict[str, List[str]]:
        assert mode in cls.dataset_modes, mode
        for subfolder in subfolders:
            assert (
                Path(root, mode, subfolder).exists()
                and Path(root, mode, subfolder).is_dir()
            ), os.path.join(root, mode, subfolder)

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
        assert (
            set(map_(len, imgs.values())).__len__() == 1
        ), "imgs list have component with different length."

        for subfolder in subfolders:
            if verbose:
                print(f"found {len(imgs[subfolder])} images in {subfolder}\t")
        return imgs


class MedicalImageSegmentationDatasetWithMetaInfo(MedicalImageSegmentationDataset):
    def __init__(
        self,
        root_dir: str,
        mode: str,
        subfolders: List[str],
        transforms: SequentialWrapper = None,
        patient_pattern: str = None,
        verbose=True,
        metainfo_generator=None,
    ) -> None:
        super().__init__(
            root_dir, mode, subfolders, transforms, patient_pattern, verbose
        )
        self.metainfo_generator = metainfo_generator
