from unittest import TestCase

from PIL import Image

from deepclustering.augment import SequentialWrapper, pil_augment
from torchvision import transforms

from deepclustering.dataset.segmentation.iSeg2017_dataset import ISeg2017Dataset, ISeg2017SemiInterface
from pathlib import Path
import shutil, os


class Test_iSegDataset(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.dataset_root = "./"
        self.dataset_subfolders = ["T1", "T2", "Labels"]

        if Path(self.dataset_root, ISeg2017Dataset.folder_name).exists():
            shutil.rmtree(Path(self.dataset_root, ISeg2017Dataset.folder_name), ignore_errors=True)
        if Path(self.dataset_root, ISeg2017Dataset.zip_name).exists():
            os.remove(Path(self.dataset_root, ISeg2017Dataset.zip_name))

    def test_dataset_iteration(self):
        dataset = ISeg2017Dataset(root_dir=self.dataset_root, subfolders=self.dataset_subfolders, verbose=True,
                              mode="train")
        for i in range(len(dataset)):
            (T1, T2, Labels), filename = dataset[i]

    def tearDown(self) -> None:
        super().tearDown()
        if Path(self.dataset_root, ISeg2017Dataset.folder_name).exists():
            shutil.rmtree(Path(self.dataset_root, ISeg2017Dataset.folder_name), ignore_errors=True)
        if Path(self.dataset_root, ISeg2017Dataset.zip_name).exists():
            os.remove(Path(self.dataset_root, ISeg2017Dataset.zip_name))

    def test_semi_dataloader(self):
        iseg_handler = ISeg2017SemiInterface(
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

        (T1, T2, Labels), filename = iter(labeled_loader).__next__()
        multi_slice_viewer_debug(T1.squeeze(), T2.squeeze(), Labels.squeeze(), block=False)

        (T1, T2, Labels), filename = iter(unlabeled_loader).__next__()
        multi_slice_viewer_debug(T1.squeeze(), T2.squeeze(), Labels.squeeze(), block=False)

        (T1, T2, Labels), filename = iter(val_loader).__next__()
        multi_slice_viewer_debug(T1.squeeze(), T2.squeeze(), Labels.squeeze(), block=True)



