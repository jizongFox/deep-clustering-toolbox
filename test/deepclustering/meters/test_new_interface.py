from unittest import TestCase

import torch
import torch.nn as nn

from deepclustering.meters import MeterInterface, AverageValueMeter, SliceDiceMeter


class TestMeterInterface(TestCase):
    def setUp(self) -> None:
        self.meter_config = {
            "loss": AverageValueMeter(),
            "tra_dice": SliceDiceMeter(C=5),
        }
        self.criterion = nn.CrossEntropyLoss()

    def _batch_generator(self, batchsize=10):
        img = torch.randn(batchsize, 5, 256, 256)  # C=5
        gt = torch.randint(0, 5, (batchsize, 256, 256)).long()
        return img, gt

    def _training_loop(self, Meter, num_batchs=10):
        for num_batch in range(num_batchs):
            pred, gt = self._batch_generator()
            loss = self.criterion(pred, gt)
            Meter["loss"].add(loss.item())
            Meter["tra_dice"].add(pred, gt)

    def test_Initialize_MeterInterface(self):
        Meter = MeterInterface(meter_config=self.meter_config)
        for epoch in range(3):
            self._training_loop(Meter)
            Meter.step()
            print(Meter.summary())

    def test_save_checkpoint_and_load(self):
        Meter1 = MeterInterface(meter_config=self.meter_config)
        for epoch in range(3):
            self._training_loop(Meter1)
            Meter1.step()
            print(Meter1.summary())
        meter1_dict = Meter1.state_dict()
        # print("Meter1 saved.")
        Meter2 = MeterInterface(meter_config=self.meter_config)
        Meter2.load_state_dict(meter1_dict)
        # print("Meter2 loaded")
        print(Meter2.summary())
        for epoch in range(5):
            self._training_loop(Meter2)
            Meter2.step()
            print(Meter2.summary())
