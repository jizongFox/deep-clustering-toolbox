import time
from unittest import TestCase

import torch
from deepclustering.loss.dice_loss import TwoDimDiceLoss, dice_batch, MetaDice, ThreeDimDiceLoss
from deepclustering.utils import class2one_hot, logit2one_hot
from torch.nn import functional as F


class TestDiceLoss(TestCase):
    def setUp(self) -> None:
        self.predict_logit = torch.randn(10, 3, 256, 256).cuda()
        self.target = torch.randint(0, 3, (10, 256, 256)).cuda()

    def test_mask_dice(self):
        iteration = 1000
        criterion = DiceLoss()
        onehot_pred = logit2one_hot(self.predict_logit)
        onehot_target = class2one_hot(self.target, 3)
        start = time.time()
        for _ in range(iteration):
            loss1 = criterion(onehot_pred.float(), onehot_target)
        end1 = time.time()
        print(f"for method 1, costed time:{end1 - start}")
        for _ in range(iteration):
            loss2 = dice_batch(onehot_pred, onehot_target)
        end2 = time.time()
        print(f"for method2 costed time{end2 - end1}")
        for _ in range(iteration):
            loss3 = MetaDice(method='3d')(onehot_pred, onehot_target, )
        end3 = time.time()
        print(f"for method3 costed time{end3 - end2}")

        assert torch.allclose(1 - loss1, loss2)
        assert torch.allclose(loss2, loss3)

    def test_dice_loss(self):
        for i in range(10):
            self.predict_logit = torch.randn(10, 3, 256, 256).cuda()

            self.target = torch.randint(0, 3, (10, 256, 256)).cuda()
            pred = F.softmax(self.predict_logit, 1)
            pred.requires_grad = True
            onehot_target = class2one_hot(self.target, 3)
            criterion = DiceLoss()
            loss1 = criterion(pred, onehot_target)

            loss2 = dice_batch(pred, onehot_target)
            loss2.mean().backward()
            assert torch.allclose(1 - loss1, loss2, rtol=1e-3, atol=1e-2)

    def test_2ddice_loss(self):
        criterion = TwoDimDiceLoss(reduce=False, ignore_index=0)
        loss, dices = criterion(F.softmax(self.predict_logit, 1), class2one_hot(self.target, 3))
        print()

    def test_3ddice_loss(self):
        criterion = ThreeDimDiceLoss()
        loss, dices = criterion(F.softmax(self.predict_logit, 1), class2one_hot(self.target, 3))
        print()
