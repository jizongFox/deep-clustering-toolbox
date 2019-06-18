import os
from unittest import TestCase

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from deepclustering import DATA_PATH
from deepclustering.augment.augment import Resize, PILCutout, RandomCrop, RandomHorizontalFlip, ToTensor, ToLabel, \
    Compose
from deepclustering.augment.sychronized_augment import SequentialWrapper
from deepclustering.dataset.segmentation.medicalSegmentationDataset import MedicalImageSegmentationDataset


class TestMedicalDataSegmentation(TestCase):
    def test_segmentation_dataset(self):
        dataset_dict = {
            'root_dir': os.path.join(DATA_PATH, 'ACDC-all'),
            'mode': 'train',
            'subfolders': ['img', 'gt'],
            'verbose': True
        }
        dataset = MedicalImageSegmentationDataset(**dataset_dict)
        data, filename = dataset[2]
        print(data[0].shape)
        print(filename)

    def test_segmentatin_dataset_with_SequentialWrapper(self):
        dataset_dict = {
            'root_dir': os.path.join(DATA_PATH, 'ACDC-all'),
            'mode': 'train',
            'subfolders': ['img', 'gt'],
            'verbose': True
        }
        img_transform = Compose([
            PILCutout(min_box=56, max_box=56),
            RandomHorizontalFlip(),
            RandomCrop((192, 192)),
            Resize((256, 256)),
            ToTensor(),
        ])
        target_transform = Compose([
            PILCutout(min_box=56, max_box=56),
            RandomHorizontalFlip(),
            RandomCrop((192, 192)),
            Resize((256, 256), interpolation=0),
            ToLabel(mapping={0: 0, 1: 1, 2: 0, 3: 0}),
        ])
        transforms = SequentialWrapper(img_transform=img_transform, target_transform=target_transform,
                                       if_is_target=[False, True])

        dataset = MedicalImageSegmentationDataset(**dataset_dict, transforms=transforms)
        dataloader = iter(DataLoader(dataset, batch_size=1, shuffle=True))
        for i in range(100):
            data, filename = dataloader.__next__()
            plt.clf()
            plt.imshow(data[0].squeeze(), cmap='gray')
            plt.contourf(data[1].squeeze(), alpha=0.1)
            plt.show(block=False)
            plt.pause(1)
