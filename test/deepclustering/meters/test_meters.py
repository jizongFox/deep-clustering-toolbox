from unittest import TestCase

import torch
import torch.nn.functional as F
from deepclustering.meters import DiceMeter
from deepclustering.utils import simplex

PREDICTION_SIZE = (2, 5, 256, 256)  # 2 images with 5 classes, h, w = 256
GT_SIZE = (2, 256, 256)


class Test_Meters(TestCase):

    def setUp(self) -> None:
        super().setUp()

        self.pred_logit = torch.randn(*PREDICTION_SIZE)
        self.pred_simplex = F.softmax(self.pred_logit, 1)
        assert simplex(self.pred_simplex)
        self.gt = torch.randint(0, PREDICTION_SIZE[1], size=GT_SIZE)
        assert torch.all(self.gt.unique() < PREDICTION_SIZE[1])

    def test_dice(self):
        dicemeter = DiceMeter(method='2d', report_axises='all', C=5)
        dicemeter.add(self.pred_logit, self.gt)
