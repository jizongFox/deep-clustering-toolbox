from unittest import TestCase

import numpy as np
import requests
import torch
from PIL import Image

from deepclustering.augment import TransformInterface

__doc__ = "this file tests functions in augment model"

URL = "https://cdn1.medicalnewstoday.com/content/images/articles/322/322868/golden-retriever-puppy.jpg"


class TestInterface(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.color_img = Image.open(requests.get(URL, stream=True).raw)
        assert np.array(self.color_img).shape[2] == 3
        self.grey_img = Image.fromarray(np.array(self.color_img)[:, :, 0])
        assert np.array(self.grey_img).shape.__len__() == 2

    def test_config1(self):
        config = {
            "randomcrop": {"size": (20, 20)},
            "resize": {"size": (32, 32)},
            "Img2Tensor": {"include_rgb": False, "include_grey": True},
        }
        transform = TransformInterface(config)
        output = transform(self.color_img)
        assert output.shape[0] == 1
        assert output.shape[1:] == torch.Size([32, 32])
        output = transform(self.grey_img)
        assert output.shape[0] == 1
        assert output.shape[1:] == torch.Size([32, 32])

    def test_config2(self):
        config = {
            "PILCutout": {"min_box": 100, "max_box": 200},
            "resize": {"size": (321, 321)},
            "Img2Tensor": {"include_rgb": True, "include_grey": False},
        }
        transform = TransformInterface(config)
        output = transform(self.color_img)
        assert output.shape[0] == 3
        assert output.shape[1:] == torch.Size([321, 321])
        with self.assertRaises(AssertionError):
            output = transform(self.grey_img)
            assert output.shape[0] == 1
            assert output.shape[1:] == torch.Size([321, 321])
