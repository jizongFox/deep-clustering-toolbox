from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from PIL import Image

plt.ion()
URL = 'https://cdn1.medicalnewstoday.com/content/images/articles/322/322868/golden-retriever-puppy.jpg'
from deepclustering.augment.tensor_augment import TensorCutout, RandomCrop, Resize, CenterCrop


class TestCasewithSetUp(TestCase):
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


class TestTensorCutout(TestCasewithSetUp):

    def test1(self):
        transform = TensorCutout(200, 500)

        for _ in range(100):
            cropped_cimg = transform(self.cimg_np)
            cropped_cimgs = transform(self.cimgs_np)

    def test_torch(self):
        transform = TensorCutout(200, 500)
        for _ in range(100):
            cropped_timg = transform(self.timg)
            cropped_timgs = transform(self.timgs)
            # plt.imshow(cropped_timgs[0][0])
            # plt.show()


class TensorRandomCrop(TestCasewithSetUp):

    def test_num(self):
        cropped_size = [(100, 2000), (64, 64), (128, 256), (256, 128)]
        for c in cropped_size:

            transform = RandomCrop(size=c, padding=10, padding_mode='constant', pad_if_needed=True)
            for _ in range(100):
                cropped_cimg = transform(self.cimg_np)
                assert cropped_cimg.shape[2:] == c

    def test_torch(self):
        cropped_size = [(100, 2000), (64, 64), (128, 256), (256, 128)]
        for c in cropped_size:
            transform = RandomCrop(size=c, padding=10, padding_mode='constant', pad_if_needed=True)
            for _ in range(100):
                cropped_cimg = transform(self.timg)
                cropped_cimgs = transform(self.timgs)
                assert cropped_cimg.shape[2:] == c
                assert cropped_cimgs.shape[2:] == c


class TestResize(TestCasewithSetUp):

    def setUp(self) -> None:
        super().setUp()

    def test_resize(self):
        resize_size = [(100, 200), (100, 500), (500, 100), (1000, 2000)]
        for c in resize_size:
            transform = Resize(size=c, interpolation='bilinear')
            r_timg = transform(self.timgs)
            assert r_timg.shape[2:] == c
            r_npimg = transform(self.cimgs_np)
            assert r_npimg.shape[2:] == c


class TestCenterCrop(TestCasewithSetUp):
    def test_centercrop(self):
        crop_size = (500, 700)
        transform = CenterCrop(size=crop_size)
        center_cimg_np = transform(self.cimg_np)
        assert center_cimg_np.shape[2:] == crop_size
        center_cimgs_np = transform(self.cimgs_np)
        assert center_cimgs_np.shape[2:] == crop_size

        center_timg = transform(self.timg)
        assert center_timg.shape[2:] == crop_size

        center_timgs = transform(self.timgs)
        assert center_timgs.shape[2:] == crop_size

    def test_raise_error(self):
        crop_size = (1500, 1700)
        transform = CenterCrop(size=crop_size)
        with self.assertRaises(AssertionError):
            center_cimg_np = transform(self.cimg_np)
            assert center_cimg_np.shape[2:] == crop_size
