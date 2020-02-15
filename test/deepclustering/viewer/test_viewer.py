from unittest import TestCase
import numpy as np
import torch
from medpy.io import load as med_load

from deepclustering.viewer import multi_slice_viewer_debug


def np2torch(input):
    assert isinstance(input, np.ndarray), type(input)
    return torch.from_numpy(input).float()


def torch2numpy(input):
    assert isinstance(input, torch.Tensor), type(input)
    return input.detach().cpu().numpy()


class TestViewer(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._t1 = np2torch(med_load("subject-9-T1.hdr")[0].transpose(1, 2, 0))
        self._t2 = np2torch(med_load("subject-9-T2.hdr")[0].transpose(1, 2, 0))
        self._l1 = np2torch(med_load("subject-9-label.hdr")[0].transpose(1, 2, 0) == 10)
        self._l2 = np2torch(
            med_load("subject-9-label.hdr")[0].transpose(1, 2, 0) == 150
        )
        self._l3 = np2torch(
            med_load("subject-9-label.hdr")[0].transpose(1, 2, 0) == 250
        )
        assert self._t1.shape == self._t2.shape == self._l1.shape

    def test_load_torch(self):
        multi_slice_viewer_debug(self._t1, block=True)  # only one image
        multi_slice_viewer_debug(
            [self._t1, self._t2], block=True
        )  # only two images on pair without given gts
        multi_slice_viewer_debug(
            self._t1, self._l1, self._l2, self._l3, block=True
        )  # only one image with 3 gts
        multi_slice_viewer_debug(
            [self._t1, self._t2], self._l1, self._l2, self._l3, block=True
        )  # two images with 3 shared gt

    def test_load_numpy(self):
        t1 = torch2numpy(self._t1)
        t2 = torch2numpy(self._t2)
        l1 = torch2numpy(self._l1)
        l2 = torch2numpy(self._l2)
        l3 = torch2numpy(self._l3)

        multi_slice_viewer_debug(t1, block=True)
        multi_slice_viewer_debug([t1, t2], block=True)
        multi_slice_viewer_debug((t1, t2), block=True)
        multi_slice_viewer_debug(t1, l1, l2, l3, block=True)
        multi_slice_viewer_debug([t1, t2], l1, l2, l3, block=True)
        multi_slice_viewer_debug((t1, t2), l1, l2, l3, block=True)
