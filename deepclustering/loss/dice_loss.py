import torch
from torch import nn, Tensor

from .loss import _check_reduction_params
from ..utils import simplex, one_hot


class MetaDice(nn.Module):
    """
    3D and 2D dice computator, not the loss
    """

    def __init__(
        self,
        method: str,
        weight: Tensor = None,
        reduce: bool = False,
        eps: float = 1e-8,
    ) -> None:
        """
        :param method must be in (2d, 3d)
        :param weight: Weight to be multipled to each class.
        :param eps: default to 1e-8
        :param reduce: if reduce classwise mean. mean on batch samples.
        :return:
        """
        super(MetaDice, self).__init__()
        assert method in ("2d", "3d"), method
        self.method = method
        self.reduce = reduce
        self.eps = eps
        self.weight = weight

    def forward(self, pred: Tensor, target: Tensor):
        """
        :param pred: softmax or one_hot prediction, with or without gradient. having the shape of B C H W etc
        :param target: One_hot mask of the target, must have the same shape as the `pred`
        :param pred:
        :param target:
        :return:
        """
        assert pred.shape == target.shape, (
            f"`pred` and `target` should have the same shape, "
            f"given `pred`:{pred.shape}, `target`:{target.shape}."
        )
        assert not target.requires_grad
        assert simplex(pred), f"pred should be simplex, given {pred}."
        assert one_hot(target), f"target should be onehot, given {target}."
        pred, target = pred.float(), target.float()

        B, C, *hw = pred.shape
        reduce_axises = (
            list(range(2, pred.shape.__len__()))
            if self.method == "2d"
            else [0] + list(range(2, pred.shape.__len__()))
        )
        intersect = (pred * target).sum(reduce_axises)
        union = (pred + target).sum(reduce_axises)

        # TODO: add the weight here.
        if self.weight is not None:
            intersect = self.weight * intersect

        dices = (2.0 * intersect + self.eps) / (union + self.eps)
        assert (
            dices.shape == torch.Size([B, C])
            if self.method == "2d"
            else torch.Size([C])
        )
        if self.reduce and self.method == "2d":
            return dices.mean(0)
        return dices


dice_coef = MetaDice(method="2d", reduce=False)
dice_batch = MetaDice(method="3d", reduce=False)  # used for 3d dice


class GeneralizedDiceLoss(nn.Module):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(
        self, epsilon=1e-8, weight=None, pow=1, ignore_index=None, reduction="mean",
    ):
        super(GeneralizedDiceLoss, self).__init__()
        _check_reduction_params(reduction)
        self.epsilon = epsilon
        self.weight = weight
        self.ignore_index = ignore_index
        self.pow = pow
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor):
        """
        input: Tensor, simplex tensor as the probability distribution
        target: Tensor, one_hot tensor
        """
        assert (
            input.size() == target.size()
        ), "'input' and 'target' must have the same shape"
        # so the target here is the onehot
        assert simplex(input) and one_hot(target)

        b, c, *hw = input.shape
        if len(hw) == 0:
            input = input.unsqueeze(-1)
            target = target.unsqueeze(-1)
        noreducedim = list(range(2, len(input.shape)))

        target = target.float()
        # mask ignore_index if present
        if self.ignore_index is not None:
            mask = target.clone().ne_(self.ignore_index)
            mask.requires_grad = False
            input = input * mask
            target = target * mask

        intersect = (input * target).sum(dim=noreducedim)
        assert intersect.shape == torch.Size([b, c]), intersect.shape
        denominator = (input.pow(self.pow) + target.pow(self.pow)).sum(dim=noreducedim)
        assert denominator.shape == torch.Size([b, c]), intersect.shape

        dices = 1 - (2 * intersect + self.epsilon) / (denominator + self.epsilon)
        if self.weight is not None:
            dices = self.weight * dices

        if self.reduction == "none":
            return dices

        if self.reduction == "mean":
            return dices.mean()
        if self.reduction == "sum":
            return dices.sum()
        return dices
