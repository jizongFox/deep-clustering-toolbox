import os
import shutil
from pathlib import Path
from unittest import TestCase

from deepclustering.dataset.segmentation.iSeg2017_dataset import (
    ISeg2017Dataset,
    ISeg2017SemiInterface,
)


class TestDownloadDataset(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.dataset_root = "./"
        self.dataset_subfolders = ["T1", "T2", "Labels"]
        if Path(self.dataset_root, ISeg2017Dataset.folder_name).exists():
            shutil.rmtree(
                Path(self.dataset_root, ISeg2017Dataset.folder_name), ignore_errors=True
            )
        if Path(self.dataset_root, ISeg2017Dataset.zip_name).exists():
            os.remove(Path(self.dataset_root, ISeg2017Dataset.zip_name))

    def test_download_dataset(self):
        dataset = ISeg2017Dataset(
            root_dir=self.dataset_root,
            subfolders=self.dataset_subfolders,
            verbose=True,
            mode="train",
        )
        assert len(dataset) == 810
        assert dataset.get_group_list().__len__() == 8

        dataset = ISeg2017Dataset(
            root_dir=self.dataset_root,
            subfolders=self.dataset_subfolders,
            verbose=True,
            mode="val",
        )
        assert len(dataset) == 200
        assert dataset.get_group_list().__len__() == 2

    def tearDown(self) -> None:
        super().tearDown()
        if Path(self.dataset_root, ISeg2017Dataset.folder_name).exists():
            shutil.rmtree(
                Path(self.dataset_root, ISeg2017Dataset.folder_name), ignore_errors=True
            )
        if Path(self.dataset_root, ISeg2017Dataset.zip_name).exists():
            os.remove(Path(self.dataset_root, ISeg2017Dataset.zip_name))


class Test_SemiDataloader(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.dataset_root = "./"
        if Path(self.dataset_root, ISeg2017Dataset.folder_name).exists():
            shutil.rmtree(
                Path(self.dataset_root, ISeg2017Dataset.folder_name), ignore_errors=True
            )
        if Path(self.dataset_root, ISeg2017Dataset.zip_name).exists():
            os.remove(Path(self.dataset_root, ISeg2017Dataset.zip_name))

    def test_semi_dataloader(self):
        iseg_handler = ISeg2017SemiInterface(
            root_dir=self.dataset_root,
            labeled_data_ratio=0.8,
            unlabeled_data_ratio=0.2,
            seed=0,
            verbose=True,
        )
        iseg_handler.compile_dataloader_params(
            batch_size=20, shuffle=True, drop_last=True
        )

        (
            labeled_loader,
            unlabeled_loader,
            val_loader,
        ) = iseg_handler.SemiSupervisedDataLoaders(
            labeled_transform=None,
            unlabeled_transform=None,
            val_transform=None,
            group_labeled=True,
            group_unlabeled=True,
            group_val=True,
        )

        from deepclustering.viewer import multi_slice_viewer_debug

        (T1, T2, Labels), filename = iter(labeled_loader).__next__()
        multi_slice_viewer_debug(T1.squeeze(), Labels.squeeze(), block=False)
        multi_slice_viewer_debug(T2.squeeze(), Labels.squeeze(), block=False)

        (T1, T2, Labels), filename = iter(unlabeled_loader).__next__()
        multi_slice_viewer_debug(
            T1.squeeze(), T2.squeeze(), Labels.squeeze(), block=False
        )

        (T1, T2, Labels), filename = iter(val_loader).__next__()
        multi_slice_viewer_debug(
            T1.squeeze(), T2.squeeze(), Labels.squeeze(), block=False
        )

    def tearDown(self) -> None:
        super().tearDown()
        if Path(self.dataset_root, ISeg2017Dataset.folder_name).exists():
            shutil.rmtree(
                Path(self.dataset_root, ISeg2017Dataset.folder_name), ignore_errors=True
            )
        if Path(self.dataset_root, ISeg2017Dataset.zip_name).exists():
            os.remove(Path(self.dataset_root, ISeg2017Dataset.zip_name))
