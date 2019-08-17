import os
from unittest import TestCase

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepclustering import DATA_PATH
from deepclustering.augment.pil_augment import (
    Resize,
    PILCutout,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
    ToLabel,
    Compose,
)
from deepclustering.augment.sychronized_augment import SequentialWrapper
from deepclustering.dataset import BackgroundGenerator
from deepclustering.dataset.segmentation.medicalSegmentationDataset import (
    MedicalImageSegmentationDataset,
)
from deepclustering.decorator.decorator import TimeBlock


class TestMedicalDataSegmentationWithBackgroundGenerator(TestCase):
    def setUp(self) -> None:
        dataset_dict = {
            "root_dir": os.path.join(DATA_PATH, "ACDC-all"),
            "mode": "train",
            "subfolders": ["img", "gt"],
            "verbose": True,
        }
        img_transform = Compose(
            [
                PILCutout(min_box=56, max_box=56),
                RandomHorizontalFlip(),
                RandomCrop((192, 192)),
                Resize((256, 256)),
                ToTensor(),
            ]
        )
        target_transform = Compose(
            [
                PILCutout(min_box=56, max_box=56),
                RandomHorizontalFlip(),
                RandomCrop((192, 192)),
                Resize((256, 256), interpolation=0),
                ToLabel(mapping={0: 0, 1: 1, 2: 0, 3: 0}),
            ]
        )
        transforms = SequentialWrapper(
            img_transform=img_transform,
            target_transform=target_transform,
            if_is_target=[False, True],
        )
        self.dataset = MedicalImageSegmentationDataset(transforms=None, **dataset_dict)

    def test_conventional_dataloader(self):
        dataloader = DataLoader(self.dataset, batch_size=8, num_workers=8)
        for _ in range(30):
            with TimeBlock() as timer:
                for i, (data, filename) in enumerate(dataloader):
                    for i in range(100):
                        _ = np.random.randn(512)

            print(timer.cost)

    def test_backgroundGenerator_dataloader(self):
        dataloader = BackgroundGenerator(
            DataLoader(self.dataset, batch_size=8, num_workers=8), max_prefetch=10
        )
        with TimeBlock() as timer:
            for i, (data, filename) in enumerate(tqdm(dataloader)):
                if i > 10:
                    break
