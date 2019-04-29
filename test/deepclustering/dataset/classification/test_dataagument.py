from unittest import TestCase

import numpy as np
import requests
from PIL import Image

from deepclustering.dataset.calssification.augment import Img2Tensor, PILCutout

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
        assert np.array(cropped_color_img).mean() < np.array(self.color_img).mean()
