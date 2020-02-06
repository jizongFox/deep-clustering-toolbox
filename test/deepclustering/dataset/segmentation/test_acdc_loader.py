import os
import shutil
from pathlib import Path
from unittest import TestCase

from deepclustering.dataset.segmentation.acdc_dataset import ACDCDataset


class TestDownloadDataset(TestCase):
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

    def test_download_dataset(self):
        dataset = ACDCDataset(
            root_dir=self.dataset_root,
            subfolders=self.dataset_subfolders,
            verbose=True,
            mode="train",
        )
        assert len(dataset) == 1674
        assert dataset.get_group_list().__len__() == 175

        dataset = ACDCDataset(
            root_dir=self.dataset_root,
            subfolders=self.dataset_subfolders,
            verbose=True,
            mode="val",
        )
        assert len(dataset) == 228
        assert dataset.get_group_list().__len__() == 25

    def test_dataset_iteration(self):
        dataset = ACDCDataset(
            root_dir=self.dataset_root,
            subfolders=self.dataset_subfolders,
            verbose=True,
            mode="train",
        )
        for i in range(len(dataset)):
            (img, gt), filename = dataset[i]

    def tearDown(self) -> None:
        super().tearDown()
        if Path(self.dataset_root, ACDCDataset.folder_name).exists():
            shutil.rmtree(
                Path(self.dataset_root, ACDCDataset.folder_name), ignore_errors=True
            )
        if Path(self.dataset_root, ACDCDataset.zip_name).exists():
            os.remove(Path(self.dataset_root, ACDCDataset.zip_name))
