import os
import re
from pathlib import Path
from typing import List, Tuple, Dict

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
        assert len(subfolders) == set(subfolders).__len__(), f"subfolders must be unique, given {subfolders}."
        assert assert_list(lambda x: isinstance(x, str), subfolders), \
            f"`subfolder` elements should be str, given {subfolders}"
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
        if self.verbose:
            print(f"->> Building {self.name}:\t")
        self.filenames = self.make_dataset(self.root_dir, self.mode, self.subfolders, verbose=verbose)
        self.debug = os.environ.get("PYDEBUG", "0") == "1"
        self.set_patient_pattern(patient_pattern)

    def __len__(self) -> int:
        if self.debug:
            return int(len(self.filenames[self.subfolders[0]]) / 10)
        return int(len(self.filenames[self.subfolders[0]]))

    def __getitem__(self, index) -> Tuple[List[Tensor], str]:
        img_list, filename_list = self._getitem_index(index)
        assert img_list.__len__() == self.subfolders.__len__()
        # make sure the filename is the same image
        assert set(map_(lambda x: Path(x).stem,
                        filename_list)).__len__() == 1, f"Check the filename list, given {filename_list}."
        filename = Path(filename_list[0]).stem
        img_list = self.transform(*img_list)
        return img_list, filename

    def _getitem_index(self, index):
        img_list = [Image.open(self.filenames[subfolder][index]) for subfolder in self.subfolders]
        filename_list = [self.filenames[subfolder][index] for subfolder in self.subfolders]
        return img_list, filename_list

    def set_patient_pattern(self, pattern: str = None):
        """
        This set patient_pattern using re library.
        :param pattern:
        :return:
        """
        self._pattern = pattern
        self._re_pattern = re.compile(self._pattern)

    def get_patient_list(self):
        if not hasattr(self, "_re_pattern"):
            raise RuntimeError("Calling `get_patient_list` before setting `set_patient_pattern`")
        return sorted(list(set([self._re_pattern.search(path).group(0) for path in self.filenames["img"]])))

    @classmethod
    def make_dataset(cls, root: str, mode: str, subfolders: List[str], verbose=True) -> Dict[str, List[str]]:
        assert mode in cls.dataset_modes
        for subfolder in subfolders:
            assert Path(root, mode, subfolder).exists() and Path(root, mode, subfolder).is_dir(), \
                os.path.join(root, mode, subfolder)

        items = [os.listdir(Path(os.path.join(root, mode, subfoloder))) for subfoloder in subfolders]
        # clear up extension
        items = sorted([[x for x in item if allow_extension(x, cls.allow_extension)] for item in items])
        assert set(map_(len, items)).__len__() == 1, map_(len, items)
        imgs = {}

        for subfolder, item in zip(subfolders, items):
            imgs[subfolder] = sorted([os.path.join(root, mode, subfolder, x_path) for x_path in item])
        assert set(map_(len, imgs.values())).__len__() == 1

        for subfolder in subfolders:
            if verbose:
                print(f"found {len(imgs[subfolder])} images in {subfolder}\t")
        return imgs


class MedicalImageSegmentationDatasetWithMetaInfo(MedicalImageSegmentationDataset):

    def __init__(self, root_dir: str, mode: str, subfolders: List[str], transforms: SequentialWrapper = None,
                 patient_pattern: str = None, verbose=True, metainfo_generator=None) -> None:
        super().__init__(root_dir, mode, subfolders, transforms, patient_pattern, verbose)
        self.metainfo_generator = metainfo_generator
