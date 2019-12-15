import os
from typing import List, Tuple

from sklearn.model_selection import train_test_split

from deepclustering.augment import SequentialWrapper
from deepclustering.dataset.segmentation.medicalSegmentationDataset import (
    MedicalImageSegmentationDataset,
)
from deepclustering.dataset.segmentation.patient_sampler import (
    SubMedicalDatasetBasedOnIndex,
)
from deepclustering.dataset.semi_helper import MedicalDatasetSemiInterface
from deepclustering import DATA_PATH


class ACDCDataset(MedicalImageSegmentationDataset):
    def __init__(
        self,
        root_dir: str,
        mode: str,
        subfolders: List[str],
        transforms: SequentialWrapper = None,
        verbose=True,
    ) -> None:
        super().__init__(
            os.path.join(root_dir, "ACDC-all"),
            mode,
            subfolders,
            transforms,
            "patient\d+_\d+_\d+",
            verbose,
        )


class ACDCSemiInterface(MedicalDatasetSemiInterface):
    def __init__(
        self,
        labeled_data_ratio: float = 0.2,
        unlabeled_data_ratio: float = 0.8,
        seed: int = 0,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            ACDCDataset,
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
        return labeled_set, unlabeled_set, val_set
