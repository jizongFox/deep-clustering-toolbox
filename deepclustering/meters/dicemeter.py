__all__ = ["SliceDiceMeter", "BatchDiceMeter"]

import torch
import torch.nn.functional as F
from torch import Tensor

from deepclustering.decorator.decorator import threaded
from ._metric import _Metric
from ..loss.dice_loss import dice_coef, dice_batch
from ..utils import probs2one_hot, class2one_hot


def toOneHot(pred_logit, mask):
    """
    :param pred_logit: logit with b,c, h, w. it is fine to pass simplex prediction or onehot.
    :param mask: gt mask with b,h,w
    :return: onehot presentation of prediction and mask, pred.shape == mask.shape == b,c, h , w
    """
    oh_predmask = probs2one_hot(F.softmax(pred_logit, 1))
    oh_mask = class2one_hot(mask.squeeze(1), C=pred_logit.shape[1])
    assert oh_predmask.shape == oh_mask.shape
    return oh_predmask, oh_mask


class _DiceMeter(_Metric):
    def __init__(self, call_function, report_axises=None, C=4) -> None:
        super().__init__()
        assert report_axises is None or isinstance(report_axises, (list, tuple))
        self.diceCall = call_function
        self.report_axis = (
            report_axises if report_axises is not None else list(range(C))
        )
        self.diceLog = []
        self.C = C

    def reset(self):
        self.diceLog = []

    def add(self, pred_logit: Tensor, gt: Tensor):
        """
        call class2one_hot to convert onehot to input.
        :param pred_logit: predicton, can be simplex or logit with shape b, c, h, w
        :param gt: ground truth label with shape b, h, w or b, 1, h, w
        :return:
        """
        assert pred_logit.shape.__len__() == 4, f"pred_logit shape:{pred_logit.shape}"
        assert gt.shape.__len__() in (3, 4)
        if gt.shape.__len__() == 4:
            assert (
                gt.shape[1] == 1
            ), f"gt shape must be 1 in the 2nd axis, given {gt.shape[1]}."
        dice_value = self.diceCall(*toOneHot(pred_logit, gt))
        if dice_value.shape.__len__() == 1:
            dice_value = dice_value.unsqueeze(0)
        assert dice_value.shape.__len__() == 2
        self.diceLog.append(dice_value)

    def value(self, **kwargs):
        log = self.log
        means = log.mean(0)
        stds = log.std(0)
        report_means = (
            log.mean(1)
            if self.report_axis == "all"
            else log[:, self.report_axis].mean(1)
        )
        report_std = report_means.std()
        report_mean = report_means.mean()
        return (report_mean, report_std), (means, stds)

    @property
    def log(self):
        try:
            log = torch.cat(self.diceLog)
        except:
            log = torch.Tensor(tuple([0 for _ in range(self.C)]))
        if len(log.shape) == 1:
            log = log.unsqueeze(0)
        assert len(log.shape) == 2
        return log

    def detailed_summary(self) -> dict:
        _, (means, _) = self.value()
        return {f"DSC{i}": means[i].item() for i in range(len(means))}

    def summary(self) -> dict:
        _, (means, _) = self.value()
        return {f"DSC{i}": means[i].item() for i in self.report_axis}


class SliceDiceMeter(_DiceMeter):
    """
    used for 2d dice for sliced input.
    """

    def __init__(self, C=4, report_axises=None) -> None:
        super().__init__(call_function=dice_coef, report_axises=report_axises, C=C)


class BatchDiceMeter(_DiceMeter):
    """
    used for 3d dice for structure input.
    """

    def __init__(self, C=4, report_axises=None) -> None:
        super().__init__(call_function=dice_batch, report_axises=report_axises, C=C)
