from unittest import TestCase

import torch
from torch.utils.data import DataLoader

from deepclustering.dataset.segmentation.toydataset import ShapesDataset, Cls_ShapesDataset, default_toy_img_transform


class TestToyExample(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.dataset = ShapesDataset(count=100,
                                     max_object_per_img=4,
                                     max_object_scale=0.25,
                                     height=100,
                                     width=100,
                                     transform=default_toy_img_transform['tf1']['img'],
                                     target_transform=default_toy_img_transform['tf1']['target'])
        self.dataloader = DataLoader(self.dataset, batch_size=4)

    def testToyDataLoader(self):
        print('dsfds')
        for i, (img, gt, instance_gt) in enumerate(self.dataloader):
            assert img.shape == torch.Size([4, 1, 128, 128])

    def test_classification(self):
        dataset = Cls_ShapesDataset()
        dataloader = DataLoader(dataset, batch_size=4)
        for i, (img, gt) in enumerate(dataloader):
            assert img.shape == torch.Size([4, 3, 256, 256])
            assert gt.shape == torch.Size([4])
