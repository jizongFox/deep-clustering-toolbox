from unittest import TestCase
from deepclustering.dataset.segmentation.wMH_dataset import WMHDataset, WMHSemiInterface
from pathlib import Path
import shutil, os


# class TestDownloadDataset(TestCase):
#
#     def setUp(self) -> None:
#         super().setUp()
#         self.dataset_root = "./"
#         self.dataset_subfolders = ["T1", "T2", "Labels"]
#         if Path(self.dataset_root, ISeg2017Dataset.folder_name).exists():
#             shutil.rmtree(Path(self.dataset_root, ISeg2017Dataset.folder_name), ignore_errors=True)
#         if Path(self.dataset_root, ISeg2017Dataset.zip_name).exists():
#             os.remove(Path(self.dataset_root, ISeg2017Dataset.zip_name))
#
#     def test_download_dataset(self):
#         dataset = ISeg2017Dataset(root_dir=self.dataset_root, subfolders=self.dataset_subfolders, verbose=True,
#                               mode="train")
#         assert len(dataset) == 1129
#         assert dataset.get_patient_list().__len__() == 40
#
#         dataset = ISeg2017Dataset(root_dir=self.dataset_root, subfolders=self.dataset_subfolders, verbose=True,
#                               mode="val")
#         assert len(dataset) == 248
#         assert dataset.get_patient_list().__len__() == 10
#
#     def tearDown(self) -> None:
#         super().tearDown()
#         if Path(self.dataset_root, ISeg2017Dataset.folder_name).exists():
#             shutil.rmtree(Path(self.dataset_root, ISeg2017Dataset.folder_name), ignore_errors=True)
#         if Path(self.dataset_root, ISeg2017Dataset.zip_name).exists():
#             os.remove(Path(self.dataset_root, ISeg2017Dataset.zip_name))


class Test_wMHDatasetDataset(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.dataset_root = "./"
        self.dataset_subfolders = ["t1", "flair", "gt"]

        if Path(self.dataset_root, WMHDataset.folder_name).exists():
            shutil.rmtree(Path(self.dataset_root, WMHDataset.folder_name), ignore_errors=True)
        if Path(self.dataset_root, WMHDataset.zip_name).exists():
            os.remove(Path(self.dataset_root, WMHDataset.zip_name))

    def test_dataset_iteration(self):
        dataset = WMHDataset(root_dir=self.dataset_root, subfolders=self.dataset_subfolders, verbose=True,
                              mode="train")
        for i in range(len(dataset)):
            (t1, flair, gt), filename = dataset[i]

    def tearDown(self) -> None:
        super().tearDown()
        if Path(self.dataset_root, WMHDataset.folder_name).exists():
            shutil.rmtree(Path(self.dataset_root, WMHDataset.folder_name), ignore_errors=True)
        if Path(self.dataset_root, WMHDataset.zip_name).exists():
            os.remove(Path(self.dataset_root, WMHDataset.zip_name))

    def test_semi_dataloader(self):
        iseg_handler = WMHSemiInterface(
            labeled_data_ratio=0.8, unlabeled_data_ratio=0.2, seed=0, verbose=True
        )
        iseg_handler.compile_dataloader_params(
            batch_size=20, shuffle=True, drop_last=True
        )

        (
            labeled_loader,
            unlabeled_loader,
            val_loader,
        ) = iseg_handler.SemiSupervisedDataLoaders(
            labeled_transform=False,
            unlabeled_transform=False,
            val_transform=False,
            group_labeled=False,
            group_unlabeled=True,
            group_val=True,
        )
        from deepclustering.viewer import multi_slice_viewer_debug

        (t1, flair, gt), filename = iter(labeled_loader).__next__()
        multi_slice_viewer_debug(t1.squeeze(), flair.squeeze(), gt.squeeze(), block=False)

        (t1, flair, gt), filename = iter(unlabeled_loader).__next__()
        multi_slice_viewer_debug(t1.squeeze(), flair.squeeze(), gt.squeeze(), block=False)

        (t1, flair, gt), filename = iter(val_loader).__next__()
        multi_slice_viewer_debug(t1.squeeze(), flair.squeeze(), gt.squeeze(), block=True)
