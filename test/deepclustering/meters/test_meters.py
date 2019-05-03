from unittest import TestCase

import pandas as pd
import torch
import torch.nn.functional as F
from deepclustering.meters import SliceDiceMeter, BatchDiceMeter, AggragatedMeter, ListAggregatedMeter, \
    AverageValueMeter
from deepclustering.utils import class2one_hot
from deepclustering.utils import simplex
from easydict import EasyDict as edict

PREDICTION_SIZE = (2, 5, 256, 256)  # 2 images with 5 classes, h, w = 256
GT_SIZE = (2, 256, 256)


class Test_DiceMeters(TestCase):

    def setUp(self) -> None:
        super().setUp()

        self.pred_logit = torch.randn(*PREDICTION_SIZE)
        self.pred_simplex = F.softmax(self.pred_logit, 1)
        assert simplex(self.pred_simplex)
        self.gt = torch.randint(0, PREDICTION_SIZE[1], size=GT_SIZE)
        assert torch.all(self.gt.unique() < PREDICTION_SIZE[1])

    def test_dice(self):
        dicemeter = BatchDiceMeter(C=5)
        dicemeter.add(self.pred_logit, self.gt.unsqueeze(1))
        with self.assertRaises(AssertionError):
            dicemeter.add(self.pred_logit, self.gt.unsqueeze(2))

    def test_onehot_as_input(self):
        diceMeter = SliceDiceMeter(C=5)
        with self.assertRaises(AssertionError):
            diceMeter.add(self.gt, self.gt)
        diceMeter.add(class2one_hot(self.gt, C=5).float(), gt=self.gt)
        assert diceMeter.value()[0][0] == 1


class Test_save_and_initialize_List_Aggregrated_Meters(TestCase):

    def setUp(self) -> None:
        super().setUp()
        METERS = edict()
        METERS.loss1 = AggragatedMeter()
        METERS.dice1 = AggragatedMeter()
        METERS.dice2 = AggragatedMeter()
        wholeMeter = ListAggregatedMeter(names=list(METERS.keys()), listAggregatedMeter=list(METERS.values()))
        self.METERS = METERS
        self.wholeMeter = wholeMeter

    def _train_loop(self, num_loop=10):
        loss1Meter = AverageValueMeter()
        slicediceMeter = SliceDiceMeter(C=5)
        batchdiceMeter = BatchDiceMeter(C=5)
        for _ in range(num_loop):
            pred_logit = torch.randn(*PREDICTION_SIZE)
            gt = torch.randint(0, PREDICTION_SIZE[1], size=GT_SIZE)
            celoss = F.cross_entropy(pred_logit, gt)
            loss1Meter.add(celoss.item())
            slicediceMeter.add(pred_logit, gt)
            batchdiceMeter.add(pred_logit, gt)
        return loss1Meter.summary(), slicediceMeter.summary(), batchdiceMeter.summary()

    def test_train(self):
        for i in range(10):
            loss1, dice1, dice2 = self._train_loop()
            for k, v in self.METERS.items():
                v.add(eval(k))
        print(self.wholeMeter.summary())
        checkpoint = self.wholeMeter.state_dict

        METERS = edict()
        METERS.loss1 = AggragatedMeter()
        METERS.dice1 = AggragatedMeter()
        METERS.dice2 = AggragatedMeter()
        wholeMeter = ListAggregatedMeter(names=list(METERS.keys()), listAggregatedMeter=list(METERS.values()))
        wholeMeter.load_state_dict(checkpoint)

    def test_load_from_cp(self):
        for i in range(5):
            loss1, dice1, dice2 = self._train_loop()
            for k, v in self.METERS.items():
                v.add(eval(k))
        print(self.wholeMeter.summary())
        checkpoint = self.wholeMeter.state_dict

        new_meter, new_whole_meter = ListAggregatedMeter.initialize_from_state_dict(checkpoint)
        assert pd.DataFrame.equals(new_meter.summary(), self.wholeMeter.summary())
        assert id(new_meter) != id(self.wholeMeter)
