import os
from pathlib import Path
from typing import List, Tuple

from deepclustering.augment import SequentialWrapper
from deepclustering.dataset.segmentation import (
    MedicalImageSegmentationDataset,
    SubMedicalDatasetBasedOnIndex,
)
from sklearn.model_selection import train_test_split
from termcolor import colored

from semi_cluster import DATA_PATH
from ..semi_helper import MedicalDatasetSemiInterface
from ...utils.download_unzip_helper import download_and_extract_archive


class SpleenDataset(MedicalImageSegmentationDataset):
    download_link = "https://drive.google.com/uc?id=1VG14fqf6EltsR7HUs5dFvN0X7ru0w_wH"
    zip_name = "Spleen.zip"
    folder_name = "Spleen"

    def __init__(
            self,
            root_dir: str,
            mode: str,
            subfolders: List[str],
            transforms: SequentialWrapper = None,
            verbose=True,
    ) -> None:
        if Path(root_dir, self.folder_name).exists() and Path(root_dir, self.folder_name).is_dir():
            print(f"Found {self.folder_name}.")
        else:
            download_and_extract_archive(url=self.download_link, download_root=root_dir, extract_root=root_dir,
                                         filename=self.zip_name, remove_finished=False)
        super().__init__(
            os.path.join(root_dir, "Spleen"),
            mode,
            subfolders,
            transforms,
            "Patient_\d+",
            verbose,
        )
        print(colored(f"{self.__class__.__name__} intialized.", "green"))








class SpleenSemiInterface(MedicalDatasetSemiInterface):
    def __init__(
            self,
            labeled_data_ratio: float = 0.2,
            unlabeled_data_ratio: float = 0.8,
            seed: int = 0,
            verbose: bool = True,
    ) -> None:
        super().__init__(
            SpleenDataset,
            DATA_PATH,
            labeled_data_ratio,
            unlabeled_data_ratio,
            seed,
            verbose,
        )

    def _create_semi_supervised_datasets(
            self,
            labeled_transform: SequentialWrapper = None,
            unlabeled_transform: SequentialWrapper = None,
            val_transform: SequentialWrapper = None,
    ) -> Tuple[
        MedicalImageSegmentationDataset,
        MedicalImageSegmentationDataset,
        MedicalImageSegmentationDataset,
    ]:
        train_set = self.DataClass(
            root_dir=self.root_dir,
            mode="train",
            subfolders=["img", "gt"],
            transforms=None,
            verbose=self.verbose,
        )
        val_set = self.DataClass(
            root_dir=self.root_dir,
            mode="val",
            subfolders=["img", "gt"],
            transforms=None,
            verbose=self.verbose,
        )

        labeled_patients, unlabeled_patients = train_test_split(
            train_set.get_patient_list(),
            test_size=self.unlabeled_ratio,
            random_state=self.seed,
        )
        labeled_set = SubMedicalDatasetBasedOnIndex(train_set, labeled_patients)
        unlabeled_set = SubMedicalDatasetBasedOnIndex(train_set, unlabeled_patients)
        assert (
                labeled_set.filenames["img"].__len__()
                + unlabeled_set.filenames["img"].__len__()
                == train_set.filenames["img"].__len__()
        ), "wrong on labeled/unlabeled split."
        del train_set
        if self.verbose:
            print(
                f"labeled_dataset:{labeled_set.get_patient_list().__len__()} Patients"
            )
            print(
                f"unlabeled_dataset:{unlabeled_set.get_patient_list().__len__()} Patients"
            )
        labeled_set.transform = labeled_transform
        unlabeled_set.transform = unlabeled_transform
        val_set.transform = val_transform
        return labeled_set, unlabeled_set, val_set
