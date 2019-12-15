from unittest import TestCase

import numpy as np
import requests
from PIL import Image

from deepclustering.augment import TransformInterface
from deepclustering.augment.sychronized_augment import SequentialWrapper

URL = f"https://cdn1.medicalnewstoday.com/content/images/articles/322/322868/golden-retriever-puppy.jpg"


class Test_Sequential_Wrapper(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.color_img = Image.open(requests.get(URL, stream=True).raw)
        assert np.array(self.color_img).shape[2] == 3
        self.mask = Image.fromarray(np.array(self.color_img)[:, :, 0])
        assert np.array(self.mask).shape.__len__() == 2

    def test_synchronized_transform(self):
        config_1 = {
            "randomcrop": {"size": (200, 200)},
            "resize": {"size": (320, 320)},
            "Img2Tensor": {"include_rgb": False, "include_grey": True},
        }
        transform1 = TransformInterface(config_1)
        transform2 = TransformInterface(config_1)
        synchronized_transform = SequentialWrapper(
            img_transform=transform1,
            target_transform=transform2,
            if_is_target=[False, False],
        )
        result_imgs = synchronized_transform(self.color_img, self.color_img)
        assert np.allclose(np.array(result_imgs[0]), np.array(result_imgs[1]))
