from typing import Union, List

import torch
from torch import Tensor

from deepclustering.meters._metric import _Metric
from deepclustering.meters import BatchDiceMeter
from deepclustering.loss.dice_loss import MetaDice
from deepclustering.utils import (
    simplex,
    one_hot,
    class2one_hot,
    probs2one_hot,
    to_float,
)
from collections.abc import Iterable
import numpy as np


class UniversalDice(_Metric):
    def __init__(self, C=4, report_axises=None) -> None:
        super(UniversalDice, self).__init__()
        assert report_axises is None or isinstance(
            report_axises, (list, tuple)
        ), f"`report_axises` should be either None or an iterator, given {type(report_axises)}"
        if report_axises is not None:
            assert max(report_axises) <= C, (
                "Incompatible parameter of `C`={} and "
                "`report_axises`={}".format(C, report_axises)
            )
        self._C = C
        self._report_axis = list(range(self._C))
        if report_axises is not None:
            self._report_axis = report_axises
        self.reset()

    def reset(self):
        self._intersections = []
        self._unions = []
        self._group_names = []
        self._n = 0

    def add(
        self, pred: Tensor, target: Tensor, group_name: Union[str, List[str]] = None
    ):
        """
        add pred and target
        :param pred: class- or onehot-coded tensor of the same shape as the target
        :param target: class- or onehot-coded tensor of the same shape as the pred
        :param group_name: List of names, or a string of a name, or None.
                        indicating 2D slice dice, batch-based dice
        :return:
        """

        assert pred.shape == target.shape, (
            f"incompatible shape of `pred` and `target`, given "
            f"{pred.shape} and {target.shape}."
        )

        assert not pred.requires_grad and not target.requires_grad

        if group_name is not None:
            if not isinstance(group_name, str):
                if isinstance(group_name, Iterable):
                    assert (
                        len(group_name) == pred.shape[0]
                    )  # number of group name should be the same as the pred batch size
                    assert isinstance(group_name[0], str)
                else:
                    raise TypeError(f"type of `group_name` wrong {type(group_name)}")

        onehot_pred, onehot_target = self._convert2onehot(pred, target)
        B, C, *hw = pred.shape

        # current group name:
        current_group_name = [
            str(self._n) + f"_{i:03d}" for i in range(B)
        ]  # make it like slice based dice
        if group_name is not None:
            current_group_name = group_name
            if isinstance(group_name, str):
                # this is too make 3D dice.
                current_group_name = [group_name] * B
        assert isinstance(current_group_name, list)
        interaction, union = (
            self._intersaction(onehot_pred, onehot_target),
            self._union(onehot_pred, onehot_target),
        )
        self._intersections.append(interaction)
        self._unions.append(union)
        self._group_names.extend(current_group_name)
        self._n += 1

    @property
    def log(self):
        if self._n > 0:
            group_names = self.group_names
            interaction_array = torch.cat(self._intersections, dim=0)
            union_array = torch.cat(self._unions, dim=0)
            group_name_array = np.asarray(self._group_names)
            resulting_dice = []
            for unique_name in group_names:
                index = group_name_array == unique_name
                group_dice = (2 * interaction_array[index].sum(0) + 1e-6) / (
                    union_array[index].sum(0) + 1e-6
                )
                resulting_dice.append(group_dice)
            resulting_dice = torch.stack(resulting_dice, dim=0)
            return resulting_dice

    def value(self, **kwargs):
        if self._n == 0:
            return ([np.nan] * self._C, [np.nan] * self._C)

        resulting_dice = self.log
        return (resulting_dice.mean(0), resulting_dice.std(0))

    def summary(self) -> dict:
        means, stds = self.value()
        return {f"DSC{i}": to_float(means[i]) for i in self._report_axis}

    def detailed_summary(self) -> dict:
        means, stds = self.value()
        return {
            **{f"DSC{i}": to_float(means[i]) for i in self._report_axis},
            **{f"DSC_std{i}": to_float(stds[i]) for i in self._report_axis},
        }

    @property
    def group_names(self):
        return sorted(set(self._group_names))

    @staticmethod
    def _intersaction(pred: Tensor, target: Tensor):
        """
        return the interaction, supposing the two inputs are onehot-coded.
        :param pred: onehot pred
        :param target: onehot target
        :return: tensor of intersaction over classes
        """
        assert pred.shape == target.shape
        assert one_hot(pred) and one_hot(target)

        B, C, *hw = pred.shape
        intersect = (pred * target).sum(list(range(2, 2 + len(hw))))
        assert intersect.shape == (B, C)
        return intersect

    @staticmethod
    def _union(pred: Tensor, target: Tensor):
        """
        return the union, supposing the two inputs are onehot-coded.
        :param pred: onehot pred
        :param target: onehot target
        :return: tensor of intersaction over classes
        """
        assert pred.shape == target.shape
        assert one_hot(pred) and one_hot(target)

        B, C, *hw = pred.shape
        union = (pred + target).sum(list(range(2, 2 + len(hw))))
        assert union.shape == (B, C)
        return union

    def _convert2onehot(self, pred: Tensor, target: Tensor):
        # only two possibility: both onehot or both class-coded.
        assert pred.shape == target.shape
        # if they are onehot-coded:
        if simplex(pred, 1) and one_hot(target):
            return probs2one_hot(pred).long(), target.long()
        # here the pred and target are labeled long
        return (
            class2one_hot(pred, self._C).long(),
            class2one_hot(target, self._C).long(),
        )

    def get_plot_names(self) -> List[str]:
        return [f"DSC{i}" for i in self._report_axis]

    def __repr__(self):
        string = f"C={self._C}, report_axis={self._report_axis}\n"
        return (
            string + "\t" + "\t".join([f"{k}:{v}" for k, v in self.summary().items()])
        )
