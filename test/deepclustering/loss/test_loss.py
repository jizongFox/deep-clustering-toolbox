import unittest

import torch
import torch.nn.functional as F
from torch import nn

from deepclustering.loss import loss
from deepclustering.loss.IID_losses import IIDLoss
from deepclustering.utils import simplex, class2one_hot, iter_average


class Test_IIC(unittest.TestCase):
    def setUp(self) -> None:
        self.x1 = F.softmax(torch.randn(1, 10), 1)
        self.x2 = F.softmax(torch.randn(1, 10), 1)
        assert simplex(self.x1)
        assert simplex(self.x2)

    def test_iic(self):
        criterion = IIDLoss(lamb=1.0)
        loss = criterion(self.x1, self.x2)
        with self.assertRaises(AssertionError):
            loss = criterion(self.x1, torch.randn(5, 10))

    def test_iic2(self):
        criterion = IIDLoss(1.0)
        loss1, _ = criterion(self.x1, self.x1)
        loss2, _ = criterion(self.x2, self.x1)
        loss3, _ = criterion(self.x1, self.x2)
        assert loss2 == loss3


class TestKLDiv(unittest.TestCase):
    def setUp(self) -> None:
        self.shape = (10, 5, 224, 224)
        self.logit = torch.randn(*self.shape, requires_grad=True)
        self.pred = F.softmax(self.logit, 1)
        self.target = torch.randint(
            low=0,
            high=self.shape[1],
            size=[self.shape[i] for i in range(self.shape.__len__()) if i != 1],
        )
        self.target_oh = class2one_hot(self.target, C=self.shape[1]).float()

    def _test_kl_equivalent(self, reduction="mean"):
        kl_criterion = nn.KLDivLoss(reduction=reduction)
        kl_loss = kl_criterion(self.pred.log(), target=self.target_oh)
        _kl_loss = loss.KL_div(reduction=reduction)(self.pred, self.target_oh)
        assert torch.isclose(
            kl_loss, _kl_loss / self.shape[1] if reduction == "mean" else _kl_loss
        )

    def test_kl_equivalent(self):
        for reduction in ("sum", "mean"):
            self._test_kl_equivalent(reduction=reduction)

    def test_entropy(self):
        random_entropy = loss.Entropy()(self.pred)
        with self.assertRaises(AssertionError):
            loss.Entropy()(self.logit)
        max_entropy = loss.Entropy()(
            torch.zeros_like(self.pred).fill_(1 / self.shape[1])
        )
        assert random_entropy <= max_entropy
        zero_entropy = loss.Entropy()(torch.Tensor([[1, 0], [0, 1]]))
        assert zero_entropy == 0


class TestJSDDiv(unittest.TestCase):
    def setUp(self) -> None:
        self.shape = (10, 5, 224, 224)
        self.logit = torch.randn(*self.shape, requires_grad=True)
        self.pred = F.softmax(self.logit, 1)
        self.target = torch.randint(
            low=0,
            high=self.shape[1],
            size=[self.shape[i] for i in range(self.shape.__len__()) if i != 1],
        )
        self.target_oh = class2one_hot(self.target, C=self.shape[1]).float()

    def test_jsd(self):
        for reduction in ("sum", "mean", "none"):
            self._test_jsd(reduction=reduction)

    def _test_jsd(self, reduction="none"):
        jsd_criterion = loss.JSD_div(reduction=reduction)
        jsd_loss1 = jsd_criterion(self.pred, self.target_oh)
        jsd_loss2 = jsd_criterion(self.target_oh, self.pred)
        assert torch.allclose(jsd_loss1, jsd_loss2)
        kl_criterion = loss.KL_div(reduction=reduction)
        mean_pred = iter_average([self.target_oh.detach(), self.pred.detach()])
        mean_pred.requires_grad = True
        assert torch.allclose(
            jsd_loss1,
            0.5
            * (
                kl_criterion(mean_pred, self.target_oh.detach())
                + kl_criterion(mean_pred, self.pred.detach())
            ),
        )

    def test_jsd_2d(self):
        jsd_2d_criterion = loss.JSD_div_2D()
        jsd_map = jsd_2d_criterion(self.pred, self.target_oh)
        assert jsd_map.shape == self.target.shape


class TestCrossEntropyLoss(unittest.TestCase):
    def setUp(self) -> None:
        self.shape = (10, 5, 224, 224)
        self.logit = torch.randn(*self.shape, requires_grad=True)
        self.pred = F.softmax(self.logit, 1)
        self.target = torch.randint(
            low=0,
            high=self.shape[1],
            size=[self.shape[i] for i in range(self.shape.__len__()) if i != 1],
        )
        self.target_oh = class2one_hot(self.target, C=self.shape[1]).float()

    def _test_ce_loss(self, reduction="none"):
        ce_criterion = loss.SimplexCrossEntropyLoss(reduction=reduction)
        system_ce_criterion = nn.CrossEntropyLoss(reduction=reduction)
        celoss = ce_criterion(self.pred, self.target_oh)
        system_celoss = system_ce_criterion(self.logit, self.target)
        assert torch.allclose(celoss, system_celoss, rtol=1e-2)

    def test_CrossEntropyLoss(self):
        for reduction in ("mean", "sum", "none"):
            self._test_ce_loss(reduction)

    def _test_kl_ce(self, reduction="mean"):
        ce_criterion = loss.SimplexCrossEntropyLoss(reduction=reduction)
        kl_criterion = loss.KL_div(reduction=reduction)
        entropy_criterion = loss.Entropy(reduction=reduction)
        random_pred = F.softmax(torch.randn_like(self.pred), 1)
        kl_loss = kl_criterion(self.pred, random_pred)
        ce_loss = ce_criterion(self.pred, random_pred)
        random_entropy = entropy_criterion(random_pred)
        torch.allclose(ce_loss, kl_loss + random_entropy)

    def test_kl_ce(self):
        for reduction in ("mean", "sum", "none"):
            self._test_kl_ce(reduction)
