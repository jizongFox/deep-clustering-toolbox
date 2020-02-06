import random
import re
from copy import deepcopy as dcp
from itertools import repeat
from pathlib import Path
from typing import List, Pattern, Dict, Callable, Match

import numpy as np
from torch.utils.data.sampler import Sampler

from deepclustering.utils import id_, map_
from ._medicalSegmentationDataset import MedicalImageSegmentationDataset

__all__ = ["PatientSampler", "SubMedicalDatasetBasedOnIndex"]


class PatientSampler(Sampler):
    def __init__(
        self,
        dataset: MedicalImageSegmentationDataset,
        grp_regex: str,
        shuffle=False,
        verbose=True,
        infinite_sampler: bool = False,
    ) -> None:
        filenames: List[str] = dataset.get_filenames()
        self.grp_regex = grp_regex
        self.shuffle: bool = shuffle
        self.shuffle_fn: Callable = (
            lambda x: random.sample(x, len(x))
        ) if self.shuffle else id_
        self._infinite_sampler = infinite_sampler
        if verbose:
            print(f"Grouping using {self.grp_regex} regex")
        grouping_regex: Pattern = re.compile(self.grp_regex)

        stems: List[str] = [
            Path(filename).stem for filename in filenames
        ]  # avoid matching the extension
        matches: List[Match] = map_(grouping_regex.match, stems)
        patients: List[str] = [match.group(0) for match in matches]

        unique_patients: List[str] = sorted(list(set(patients)))
        assert len(unique_patients) < len(filenames)
        if verbose:
            print(
                f"Found {len(unique_patients)} unique patients out of {len(filenames)} images"
            )
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
        if not self._infinite_sampler:
            return self._one_iter()
        return self._infinite_iter()

    def _one_iter(self):
        values = list(self.idx_map.values())
        shuffled = self.shuffle_fn(values)
        return iter(shuffled)

    def _infinite_iter(self):

        while True:
            yield from self._one_iter()


def SubMedicalDatasetBasedOnIndex(
    dataset: MedicalImageSegmentationDataset, group_list
) -> MedicalImageSegmentationDataset:
    """
    This class divide a list of file path to some different groups in order to split the dataset based on p_pattern string.
    """
    assert (
        isinstance(group_list, (tuple, list)) and group_list.__len__() >= 1
    ), f"group_list to be extracted: {group_list}"
    dataset = dcp(dataset)
    patient_img_list: List[str] = dataset.get_filenames()
    sub_patient_index = [
        dataset._get_group_name(f) in group_list for f in patient_img_list
    ]
    dataset._filenames = {
        k: np.array(v)[np.array(sub_patient_index)].tolist()
        for k, v in dataset._filenames.items()
    }
    return dataset
