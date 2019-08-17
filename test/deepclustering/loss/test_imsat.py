from unittest import TestCase

import torch

from deepclustering.loss.IMSAT_loss import MultualInformaton_IMSAT


class TestIMSATLoss(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.pred_log = torch.randn(200, 10)

    def test_multinformation_imsat(self):
        criterion = MultualInformaton_IMSAT(mu=1.0)
        MI, _ = criterion(self.pred_log)
        assert MI > 0, f"MI should be aways positive, given {MI.item()}"
