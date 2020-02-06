import os
from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import train_test_split

from deepclustering import DATA_PATH
from deepclustering.augment import SequentialWrapper
from deepclustering.dataset.segmentation._medicalSegmentationDataset import (
    MedicalImageSegmentationDataset,
)
from deepclustering.dataset.segmentation._patient_sampler import (
    SubMedicalDatasetBasedOnIndex,
)
from deepclustering.dataset.semi_helper import MedicalDatasetSemiInterface
from deepclustering.utils.download_unzip_helper import download_and_extract_archive


class ACDCDataset(MedicalImageSegmentationDataset):
    download_link = "https://drive.google.com/uc?id=147xICU__T23aOYkdjGSA5Hh8W1SK-y9p"
    zip_name = "ACDC-all.zip"
    folder_name = "ACDC-all"

    def __init__(
        self,
        root_dir: str,
        mode: str,
        subfolders: List[str],
        transforms: SequentialWrapper = None,
        verbose=True,
    ) -> None:
        if (
            Path(root_dir, self.folder_name).exists()
            and Path(root_dir, self.folder_name).is_dir()
        ):
            print(f"Found {self.folder_name}.")
        else:
            download_and_extract_archive(
                url=self.download_link,
                download_root=root_dir,
                extract_root=root_dir,
                filename=self.zip_name,
                remove_finished=False,
            )
        super().__init__(
            os.path.join(root_dir, self.folder_name),
            mode,
            subfolders,
            transforms,
            "patient\d+_\d+_\d+",
            verbose,
        )


class ACDCSemiInterface(MedicalDatasetSemiInterface):
    def __init__(
        self,
        root_dir=DATA_PATH,
        labeled_data_ratio: float = 0.2,
        unlabeled_data_ratio: float = 0.8,
        seed: int = 0,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            ACDCDataset,
            root_dir,
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
            train_set.get_group_list(),
            test_size=self.unlabeled_ratio,
            random_state=self.seed,
        )
        labeled_set = SubMedicalDatasetBasedOnIndex(train_set, labeled_patients)
        unlabeled_set = SubMedicalDatasetBasedOnIndex(train_set, unlabeled_patients)
        assert len(labeled_set) + len(unlabeled_set) == len(
            train_set
        ), "wrong on labeled/unlabeled split."
        del train_set
        if labeled_transform:
            labeled_set.set_transform(labeled_transform)
        if unlabeled_transform:
            unlabeled_set.set_transform(unlabeled_transform)
        if val_transform:
            val_set.set_transform(val_transform)
        return labeled_set, unlabeled_set, val_set
