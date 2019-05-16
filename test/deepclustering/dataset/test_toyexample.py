from unittest import TestCase
from deepclustering.dataset.segmentation.toydataset import ShapesDataset, Cls_ShapesDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch

class TestToyExample(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.dataset = ShapesDataset(count=100, max_object_per_img=4, max_object_scale=0.25)
        self.dataloader = DataLoader(self.dataset, batch_size=4)

    def testToyDataLoader(self):
        print('dsfds')
        for i, (img, gt, instance_gt) in enumerate(self.dataloader):
            assert img.shape == torch.Size([4, 3, 256, 256])

    def test_classification(self):
        dataset = Cls_ShapesDataset()
        dataloader = DataLoader(dataset, batch_size=4)
        for i, (img, gt) in enumerate(dataloader):
            assert img.shape == torch.Size([4, 3, 256, 256])
            assert gt.shape == torch.Size([4])

