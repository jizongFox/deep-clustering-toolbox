import torch
from deepclustering.meters import (
    HaussdorffDistance,
    BatchDiceMeter,
    SliceDiceMeter,
    MeterInterface,
)
from deepclustering.utils import logit2one_hot, class2one_hot
from unittest import TestCase


class TestHaussdorffDistance(TestCase):
    def setUp(self) -> None:
        super().setUp()
        C = 3
        meter_config = {
            "hd_meter": HaussdorffDistance(C=C),
            "s_dice": SliceDiceMeter(C=C, report_axises=[1, 2]),
            "b_dice": BatchDiceMeter(C=C, report_axises=[1, 2]),
        }
        self.meter = MeterInterface(meter_config)

    def test_batch_case(self):
        print(self.meter.summary())

        for _ in range(5):
            for _ in range(10):
                pred = torch.randn(4, 3, 256, 256)
                label = torch.randint(0, 3, (4, 256, 256))

                pred_onehot = logit2one_hot(pred)
                label_onehot = class2one_hot(label, C=3)
                self.meter.hd_meter.add(pred_onehot, label_onehot)
                self.meter.s_dice.add(pred, label)
                self.meter.b_dice.add(pred, label)
            self.meter.step()
            print(self.meter.summary())
