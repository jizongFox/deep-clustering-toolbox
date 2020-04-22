from typing import List, Union

import numpy as np
from deepclustering.meters._metric import _Metric
from deepclustering.utils import (
    simplex,
    one_hot,
    class2one_hot,
    probs2one_hot,
    to_float,
)
from torch import Tensor

from .surface_distance import (
    mod_hausdorff_distance,
    hausdorff_distance,
    average_surface_distance,
)


class SurfaceMeter(_Metric):
    meter_choices = {
        "mod_hausdorff": mod_hausdorff_distance,
        "hausdorff": hausdorff_distance,
        "average_surface": average_surface_distance,
    }
    abbr = {"mod_hausdorff": "MHD", "hausdorff": "HD", "average_surface": "ASD"}

    def __init__(self, C=4, report_axises=None, metername: str = "hausdorff") -> None:
        super(SurfaceMeter, self).__init__()
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
        assert metername in self.meter_choices.keys()
        self._surface_name = metername
        self._abbr = self.abbr[metername]
        self._surface_function = self.meter_choices[metername]
        self.reset()

    def reset(self):
        self._mhd = []
        self._n = 0

    def add(
        self,
        pred: Tensor,
        target: Tensor,
        voxelspacing: Union[List[float], float] = None,
    ):
        """
        add pred and target
        :param pred: class- or onehot-coded tensor of the same shape as the target
        :param target: class- or onehot-coded tensor of the same shape as the pred
        : res: resolution for different dimension
        :return:
        """
        assert pred.shape == target.shape, (
            f"incompatible shape of `pred` and `target`, given "
            f"{pred.shape} and {target.shape}."
        )
        assert not pred.requires_grad and not target.requires_grad

        onehot_pred, onehot_target = self._convert2onehot(pred, target)
        B, C, *hw = pred.shape
        mhd = self._evalue(onehot_pred, onehot_target, voxelspacing)
        assert mhd.shape == (B, len(self._report_axis))
        self._mhd.append(mhd)
        self._n += 1

    def value(self, **kwargs):
        if self._n == 0:
            return ([np.nan] * self._C, [np.nan] * self._C)
        mhd = np.concatenate(self._mhd, axis=0)
        return (mhd.mean(0), mhd.std(0))

    def summary(self) -> dict:
        means, stds = self.value()
        return {
            f"{self._abbr}{i}": to_float(means[num])
            for num, i in enumerate(self._report_axis)
        }

    def detailed_summary(self) -> dict:
        means, stds = self.value()
        return {
            **{
                f"{self._abbr}{i}": to_float(means[num])
                for num, i in enumerate(self._report_axis)
            },
            **{
                f"{self._abbr}{i}": to_float(stds[num].item())
                for num, i in enumerate(self._report_axis)
            },
        }

    def _evalue(self, pred: Tensor, target: Tensor, voxelspacing):
        """
        return the B\times C list
        :param pred: onehot pred
        :param target: onehot target
        :return: tensor of size B x C of type np.array
        """
        assert pred.shape == target.shape
        assert one_hot(pred, axis=1) and one_hot(target, axis=1)
        B, C, *hw = pred.shape
        result = np.zeros([B, len(self._report_axis)])
        for b, (one_batch_img, one_batch_gt) in enumerate(zip(pred, target)):
            for c, (one_slice_img, one_slice_gt) in enumerate(
                zip(one_batch_img[self._report_axis], one_batch_gt[self._report_axis])
            ):
                mhd = self._surface_function(
                    one_slice_img, one_slice_gt, voxelspacing=voxelspacing
                )
                result[b, c] = mhd
        return result

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
        return [f"{self._abbr}{i}" for num, i in enumerate(self._report_axis)]

    def __repr__(self):
        string = f"C={self._C}, report_axis={self._report_axis}\n"
        return (
            string + "\t" + "\t".join([f"{k}:{v}" for k, v in self.summary().items()])
        )
