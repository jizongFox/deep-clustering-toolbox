from unittest import TestCase
from torchvision.models.segmentation import deeplabv3_resnet101
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from deepclustering.viewer import multi_slice_viewer_debug

from deepclustering.utils.adversarial_generator import FSGMGenerator


class fakenetwork:
    def __init__(self, network) -> None:
        super().__init__()
        self._network = network

    def __call__(self, *args, **kwargs):
        return self._network(*args, **kwargs)["out"]

    def zero_grad(self):
        return self._network.zero_grad()

    def eval(self):
        return self._network.eval()

    def load_state_dict(self, state_dict):
        return self._network.load_state_dict(state_dict)


class TestAdversarialFSGMGenerator(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._img = ToTensor()(Image.open("img1.jpg")).unsqueeze(0)
        assert self._img.shape[0] == 1 and self._img.shape[1] == 3
        self._network = fakenetwork(deeplabv3_resnet101(aux_loss=True, pretrained=True))
        self._network.eval()

    def test_gt(self):
        preds = self._network(self._img)
        multi_slice_viewer_debug(
            self._img.transpose(1, 3).transpose(1, 2),
            preds.max(1)[1],
            block=False,
            no_contour=False,
        )
        from copy import deepcopy as dcp

        preds2 = dcp(self._network)(self._img)
        multi_slice_viewer_debug(
            self._img.transpose(1, 3).transpose(1, 2),
            preds2.max(1)[1],
            block=False,
            no_contour=False,
        )
        assert torch.allclose(preds, preds2)

        fsgmGenerator = FSGMGenerator(net=self._network, eplision=0.01)
        adv_img, _ = fsgmGenerator(self._img, gt=None)
        assert adv_img.shape == self._img.shape
        preds_adv = self._network(adv_img)

        multi_slice_viewer_debug(
            self._img.transpose(1, 3).transpose(1, 2),
            preds_adv.max(1)[1],
            block=False,
            no_contour=False,
        )

        rand_pred = self._network(self._img + torch.randn_like(self._img).sign() * 0.01)
        multi_slice_viewer_debug(
            self._img.transpose(1, 3).transpose(1, 2),
            rand_pred.max(1)[1],
            block=True,
            no_contour=False,
        )
