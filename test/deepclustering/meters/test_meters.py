from unittest import TestCase
from deepclustering.meters import AverageValueMeter, MeterInterface, SliceDiceMeter
import numpy as np
import torch


class TestBasicInterface(TestCase):
    def test_averagevalueMeter(self):
        meter = AverageValueMeter()
        for i in range(10000):
            meter.add(1)
        print(meter)

    def test_meter_interface(self):
        meterinterface = MeterInterface(
            {"avg1": AverageValueMeter(), "dice1": SliceDiceMeter()}
        )
        print(meterinterface.summary())
        for epoch in range(10):
            if epoch == 2:
                meterinterface.register_new_meter("avg2", AverageValueMeter())
            for i in range(10):
                meterinterface["avg1"].add(1)
                meterinterface["dice1"].add(
                    torch.randn(1, 4, 224, 224), torch.randint(0, 4, size=(1, 224, 224))
                )
                try:
                    meterinterface["avg2"].add(2)
                except:
                    pass
            meterinterface.step()
        print(meterinterface.summary())

    def test_resume(self):

        meterinterface = MeterInterface(
            {"avg1": AverageValueMeter(), "dice1": SliceDiceMeter()}
        )
        meterinterface.step()
        meterinterface.step()
        meterinterface.step()

        for epoch in range(10):
            if epoch == 2:
                meterinterface.register_new_meter("avg2", AverageValueMeter())
            for i in range(10):
                meterinterface["avg1"].add(1)
                meterinterface["dice1"].add(
                    torch.randn(1, 4, 224, 224), torch.randint(0, 4, size=(1, 224, 224))
                )
                try:
                    meterinterface["avg2"].add(2)
                except:
                    pass
            meterinterface.step()
        print(meterinterface.summary())
        state_dict = meterinterface.state_dict()

        meterinterface2 = MeterInterface(
            {
                "avg1": AverageValueMeter(),
                "avg2": AverageValueMeter(),
                "dice1": SliceDiceMeter(),
                "avg3": AverageValueMeter(),
            }
        )
        meterinterface2.load_state_dict(state_dict)

        for epoch in range(10):
            for i in range(10):
                meterinterface2["avg3"].add(1)
                meterinterface2["dice1"].add(
                    torch.randn(1, 4, 224, 224), torch.randint(0, 4, size=(1, 224, 224))
                )
            meterinterface2.step()
        print(meterinterface2.summary())
