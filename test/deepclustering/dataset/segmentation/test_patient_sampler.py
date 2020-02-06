import os
import shutil
from pathlib import Path
from unittest import TestCase

from torch.utils.data import DataLoader

from deepclustering.dataset import ACDCDataset
from deepclustering.dataset import PatientSampler


class TestPatientSampler(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.dataset_root = "./"
        self.dataset_subfolders = ["img", "gt"]
        if Path(self.dataset_root, ACDCDataset.folder_name).exists():
            shutil.rmtree(
                Path(self.dataset_root, ACDCDataset.folder_name), ignore_errors=True
            )
        if Path(self.dataset_root, ACDCDataset.zip_name).exists():
            os.remove(Path(self.dataset_root, ACDCDataset.zip_name))

    def test_acdc_sampler(self):
        dataset = ACDCDataset(
            root_dir=self.dataset_root, mode="train", subfolders=self.dataset_subfolders
        )
        patient_sampler = PatientSampler(
            dataset=dataset,
            grp_regex=dataset._pattern,
            shuffle=False,
            infinite_sampler=False,
        )
        dataloader = DataLoader(dataset, batch_sampler=patient_sampler)
        for i, (_, filename) in enumerate(dataloader):
            print(filename)

    def test_infinit_sampler(self):
        dataset = ACDCDataset(
            root_dir=self.dataset_root, mode="train", subfolders=self.dataset_subfolders
        )
        patient_sampler = PatientSampler(
            dataset=dataset,
            grp_regex=dataset._pattern,
            shuffle=False,
            infinite_sampler=True,
        )
        dataloader = DataLoader(dataset, batch_sampler=patient_sampler)
        for i, (_, filename) in enumerate(dataloader):
            print(filename)
            if i == 100:
                break

    def tearDown(self) -> None:
        super().tearDown()
        if Path(self.dataset_root, ACDCDataset.folder_name).exists():
            shutil.rmtree(
                Path(self.dataset_root, ACDCDataset.folder_name), ignore_errors=True
            )
        if Path(self.dataset_root, ACDCDataset.zip_name).exists():
            os.remove(Path(self.dataset_root, ACDCDataset.zip_name))
