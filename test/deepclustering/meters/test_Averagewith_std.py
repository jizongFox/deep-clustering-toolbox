import time
from unittest import TestCase

import numpy as np
import torch

from deepclustering.meters import AveragewithStd, MeterInterface
from deepclustering.writer.draw_csv import DrawCSV2


class TestDrawAverageWithSTD(TestCase):
    """
    This is to test the plotting of mean and std of a list of varying scalars
    """

    def setUp(self) -> None:
        config = {"avg": AveragewithStd()}
        self.METER = MeterInterface(config)
        columns_to_draw = [["avg_mean", "avg_lstd", "avg_hstd"]]
        from pathlib import Path
        self.drawer = DrawCSV2(columns_to_draw=columns_to_draw, save_dir=Path(__file__).parent)

    def _train_loop(self, data, epoch):
        for i in data:
            self.METER["avg"].add(i)

        time.sleep(0.1)

    def test_torch(self):
        for i in range(100):
            data = torch.randn(10, 1) / (i + 1)
            self._train_loop(data, i)
            self.METER.step()
            summary = self.METER.summary()
            self.drawer.draw(summary)

    def test_numpy(self):
        for i in range(100):
            data = np.random.randn(10, 1) / (i + 1)
            self._train_loop(data, i)
            self.METER.step()
            summary = self.METER.summary()
            self.drawer.draw(summary)

    def test_list(self):
        for i in range(100):
            data = (np.random.randn(10, 1) / (i + 1)).squeeze().tolist()
            self._train_loop(data, i)
            self.METER.step()
            summary = self.METER.summary()
            self.drawer.draw(summary)
