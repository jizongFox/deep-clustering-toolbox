__all__ = ["PatientSampler"]
import random
import re
from itertools import repeat
from pathlib import Path
from typing import List, Pattern, Dict, Callable, Match

from torch.utils.data.sampler import Sampler

from deepclustering.utils import id_, map_
from .medicalSegmentationDataset import MedicalImageSegmentationDataset


class PatientSampler(Sampler):
    def __init__(self, dataset: MedicalImageSegmentationDataset, grp_regex: str, shuffle=False, verbose=True) -> None:
        filenames: List[str] = dataset.filenames[dataset.subfolders[0]]
        self.grp_regex = grp_regex
        self.shuffle: bool = shuffle
        self.shuffle_fn: Callable = (lambda x: random.sample(x, len(x))) if self.shuffle else id_
        if verbose:
            print(f"Grouping using {self.grp_regex} regex")
        grouping_regex: Pattern = re.compile(self.grp_regex)

        stems: List[str] = [Path(filename).stem for filename in filenames]  # avoid matching the extension
        matches: List[Match] = map_(grouping_regex.match, stems)
        patients: List[str] = [match.group(0) for match in matches]

        unique_patients: List[str] = list(set(patients))
        assert len(unique_patients) < len(filenames)
        if verbose:
            print(f"Found {len(unique_patients)} unique patients out of {len(filenames)} images")
        self.idx_map: Dict[str, List[int]] = dict(zip(unique_patients, repeat(None)))
        for i, patient in enumerate(patients):
            if not self.idx_map[patient]:
                self.idx_map[patient] = []
            self.idx_map[patient] += [i]
        assert sum(len(self.idx_map[k]) for k in unique_patients) == len(filenames)
        if verbose:
            print("Patient to slices mapping done")

    def __len__(self):
        return len(self.idx_map.keys())

    def __iter__(self):
        values = list(self.idx_map.values())
        shuffled = self.shuffle_fn(values)
        return iter(shuffled)


class ExtractPatientCut:
    """
    This class divide a list of file path to some different groups in order to split the dataset based on p_pattern string.
    """

    def __init__(self, p_pattern: str = None) -> None:
        super().__init__()
