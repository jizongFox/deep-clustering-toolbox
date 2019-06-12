#
#
#   file description
#
#
import requests
from PIL import Image
import numpy as np
from unittest import TestCase
import torch
from deepclustering.augment.tensor_augment import RandomCrop, RandomHorizontalFlip, RandomVerticalFlip

URL = f"https://cdn1.medicalnewstoday.com/content/images/articles/322/322868/golden-retriever-puppy.jpg"


class TestTensorAugmentation(TestCase):
    def setUp(self) -> None:
        self.color_img = Image.open(requests.get(URL, stream=True).raw)
        assert np.array(self.color_img).shape[2] == 3
        self.grey_img = Image.fromarray(np.array(self.color_img)[:, :, 0])
        assert np.array(self.grey_img).shape.__len__() == 2
        self.cimg_np = np.array(self.color_img).transpose((2, 0, 1))[None]
        assert self.cimg_np.shape.__len__() == 4
        self.cimgs_np = np.concatenate((self.cimg_np, self.cimg_np), 0)
        assert self.cimgs_np.shape.__len__() == 4
        assert self.cimgs_np.shape[0] == 2
        self.timg = torch.Tensor(self.cimg_np).float()
        self.timgs = torch.Tensor(self.cimgs_np).float()
        assert self.timg.shape.__len__() == 4
        assert self.timgs.shape.__len__() == 4

    def test_RandomCrop(self):
        size = (500, 300)
        transform = RandomCrop(size=size)
        r_img = transform(self.timg)
        r_imgs = transform(self.timgs)
        assert r_imgs.shape[2:] == torch.Size(size)
        assert r_img.shape[2:] == torch.Size(size)

    def test_RandmHorizontalFlip(self):
        transform = RandomHorizontalFlip(p=1)
        r_img = transform(self.timg)

    def test_RandmVertialFlip(self):
        transform = RandomVerticalFlip(p=1)
        r_img = transform(self.timg)
