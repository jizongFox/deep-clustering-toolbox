from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ._metric import _Metric
from deepclustering2.loss.dice_loss import dice_coef, dice_batch
from deepclustering2.utils import probs2one_hot, class2one_hot
from deepclustering2.utils.typecheckconvert import to_float

__all__ = ["SliceDiceMeter", "BatchDiceMeter"]


# from deepclustering.decorator.decorator import threaded


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
    def __init__(self, call_function, C=4, report_axises=None) -> None:
        super(_DiceMeter, self).__init__()
        assert report_axises is None or isinstance(report_axises, (list, tuple))
        if report_axises is not None:
            assert max(report_axises) <= C, (
                "Incompatible parameter of `C`={} and "
                "`report_axises`={}".format(C, report_axises)
            )
        self._C = C
        self._report_axis = list(range(self._C))
        if report_axises is not None:
            self._report_axis = report_axises
        self._diceCallFunction = call_function
        self._diceLog = []  # type: ignore
        self._n = 0

    def reset(self):
        self._diceLog = []  # type: ignore
        self._n = 0

    def add(self, pred_logit: Tensor, gt: Tensor):
        """
        call class2one_hot to convert onehot to input.
        :param pred_logit: predicton, can be simplex or logit with shape b, c, h, w
        :param gt: ground truth label with shape b, h, w or b, 1, h, w
        :return:
        """
        assert pred_logit.shape.__len__() == 4, f"pred_logit shape:{pred_logit.shape}"
        if gt.shape.__len__() == 4:
            gt = gt.squeeze(2)
        assert gt.shape.__len__() == 3
        dice_value = self._diceCallFunction(*toOneHot(pred_logit, gt))
        if dice_value.shape.__len__() == 1:
            dice_value = dice_value.unsqueeze(0)
        assert dice_value.shape.__len__() == 2
        self._diceLog.append(dice_value)
        self._n += 1

    def value(self):
        if self._n > 0:
            log = torch.cat(self._diceLog)
            means = log.mean(0)
            stds = log.std(0)
            report_means = log[:, self._report_axis].mean(1)
            report_std = report_means.std()
            report_mean = report_means.mean()
            return (report_mean, report_std), (means, stds)
        else:
            return (np.nan, np.nan), ([np.nan] * self._C, [np.nan] * self._C)

    def detailed_summary(self) -> dict:
        _, (means, _) = self.value()
        return {f"DSC{i}": to_float(means[i]) for i in range(len(means))}

    def summary(self) -> dict:
        _, (means, _) = self.value()
        return {f"DSC{i}": to_float(means[i]) for i in self._report_axis}

    def get_plot_names(self) -> List[str]:
        return [f"DSC{i}" for i in self._report_axis]

    def __repr__(self):
        string = f"C={self._C}, report_axis={self._report_axis}\n"
        return (
            string + "\t" + "\t".join([f"{k}:{v}" for k, v in self.summary().items()])
        )


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
