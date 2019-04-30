from unittest import TestCase

import numpy as np
import requests
from PIL import Image

from deepclustering.augment.augment import Img2Tensor, PILCutout, RandomCrop, CenterCrop

url = 'https://cdn1.medicalnewstoday.com/content/images/articles/322/322868/golden-retriever-puppy.jpg'


class TestGrey2tensor(TestCase):
    def setUp(self) -> None:
        self.color_img = Image.open(requests.get(url, stream=True).raw)
        assert np.array(self.color_img).shape[2] == 3
        self.grey_img = Image.fromarray(np.array(self.color_img)[:, :, 0])
        assert np.array(self.grey_img).shape.__len__() == 2

    def test_Img2Tensor(self):
        self.transform = Img2Tensor(include_rgb=True, include_grey=True)
        color_img_tensor = self.transform(self.color_img)
        assert color_img_tensor.shape[0] == 4
        self.transform = Img2Tensor(include_rgb=True, include_grey=False)
        color_img_tensor = self.transform(self.color_img)
        assert color_img_tensor.shape[0] == 3

        self.transform = Img2Tensor(include_rgb=False, include_grey=True)
        color_img_tensor = self.transform(self.color_img)
        assert color_img_tensor.shape[0] == 1

    def test_grey2tensor(self):
        self.transform = Img2Tensor(include_rgb=True, include_grey=True)
        grey_img_tensor = self.transform(self.grey_img)
        assert grey_img_tensor.shape[0] == 1

        self.transform = Img2Tensor(include_rgb=False, include_grey=True)
        grey_img_tensor = self.transform(self.grey_img)
        assert grey_img_tensor.shape[0] == 1

        with self.assertRaises(AssertionError):
            self.transform = Img2Tensor(include_rgb=True, include_grey=False)
            grey_img_tensor = self.transform(self.grey_img)
            assert grey_img_tensor.shape[0] == 1


class TestPILCutout(TestCase):
    def setUp(self) -> None:
        self.color_img = Image.open(requests.get(url, stream=True).raw)
        assert np.array(self.color_img).shape[2] == 3
        self.grey_img = Image.fromarray(np.array(self.color_img)[:, :, 0])
        assert np.array(self.grey_img).shape.__len__() == 2

    def test_Img2Tensor(self):
        self.transform = PILCutout(min_box=10, max_box=100)
        cropped_color_img = self.transform(self.color_img)
        assert cropped_color_img.size == self.color_img.size
        assert np.array(cropped_color_img).mean() < np.array(self.color_img).mean()  # type: ignore


class Test_RandomCrop(TestCase):
    def setUp(self) -> None:
        self.color_img = Image.open(requests.get(url, stream=True).raw)
        assert np.array(self.color_img).shape[2] == 3
        self.grey_img = Image.fromarray(np.array(self.color_img)[:, :, 0])
        assert np.array(self.grey_img).shape.__len__() == 2

    def test_random_Crop(self):
        crop_size = (256, 512)
        self.transform = RandomCrop(crop_size, padding_mode='symmetric')
        cropped_cimg = self.transform(self.color_img)
        cropped_gimg = self.transform(self.grey_img)
        assert tuple(x for x in crop_size[::-1]) == cropped_gimg.size
        assert tuple(x for x in crop_size[::-1]) == cropped_cimg.size


class TestCenterCrop(TestCase):
    def setUp(self) -> None:
        self.color_img = Image.open(requests.get(url, stream=True).raw)
        assert np.array(self.color_img).shape[2] == 3
        self.grey_img = Image.fromarray(np.array(self.color_img)[:, :, 0])
        assert np.array(self.grey_img).shape.__len__() == 2

    def test_center_crop(self):
        crop_size = (256, 512)
        self.transform = CenterCrop(crop_size)
        ccimg = self.transform(self.color_img)
        cgimg = self.transform(self.grey_img)
        assert tuple(x for x in crop_size[::-1]) == ccimg.size
        assert tuple(x for x in crop_size[::-1]) == cgimg.size

    def test_center_crop(self):
        crop_size = 256
        self.transform = CenterCrop(crop_size)
        ccimg = self.transform(self.color_img)
        cgimg = self.transform(self.grey_img)
        assert tuple([crop_size, crop_size]) == ccimg.size
        assert tuple([crop_size, crop_size]) == cgimg.size
