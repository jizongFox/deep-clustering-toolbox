from unittest import TestCase

import numpy as np
import torch

from deepclustering.decorator import TimeBlock
from deepclustering.meters import MeterInterface, AverageValueMeter, SliceDiceMeter
from deepclustering.writer._dataframedrawercallback import singleline_plot
from deepclustering.writer.dataframedrawer import DataFrameDrawer

save_name1 = "1.png"
save_name2 = "2.png"


class TestDataFrameDrawer(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._meter_config = {
            "avg1": AverageValueMeter(),
            "dice1": SliceDiceMeter(C=2),
            "dice2": SliceDiceMeter(C=2),
        }
        self.meters = MeterInterface(self._meter_config)

    def _train_loop(self):
        for i in range(2):
            scalar1 = np.random.rand()
            self.meters["avg1"].add(scalar1)
            img_pred = torch.randn(1, 2, 100, 100)
            img_gt = torch.randint(0, 2, (1, 100, 100))
            self.meters["dice1"].add(img_pred, img_gt)
            self.meters["dice2"].add(img_pred, img_gt)

    def test_plot(self):
        self.drawer = DataFrameDrawer(self.meters, save_dir="./", save_name=save_name1)
        self.meters.reset()
        for epoch in range(5):
            self._train_loop()
            self.meters.step()
            with TimeBlock() as timer:
                self.drawer()
            print(timer.cost)

    def test_single_line_plot(self):
        self.drawer = DataFrameDrawer(self.meters, save_dir="./", save_name=save_name2)
        self.meters.reset()
        self.drawer.set_callback("dice2", singleline_plot())
        for epoch in range(5):
            self._train_loop()
            self.meters.step()
            with TimeBlock() as timer:
                self.drawer()
            print(timer.cost)
