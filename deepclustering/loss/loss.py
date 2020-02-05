import torch
import torch.nn as nn
from torch import Tensor

from ..utils.general import simplex, assert_list


def _check_reduction_params(reduction):
    assert reduction in (
        "mean",
        "sum",
        "none",
    ), "reduction should be in ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``, given {}".format(
        reduction
    )


class Entropy(nn.Module):
    r"""General Entropy interface

    the definition of Entropy is - \sum p(xi) log (p(xi))

    reduction (string, optional): Specifies the reduction to apply to the output:
    ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
    ``'mean'``: the sum of the output will be divided by the number of
    elements in the output, ``'sum'``: the output will be summed.
    """

    def __init__(self, reduction="mean", eps=1e-16):
        super().__init__()
        _check_reduction_params(reduction)
        self._eps = eps
        self._reduction = reduction

    def forward(self, input: Tensor) -> Tensor:
        assert input.shape.__len__() >= 2
        b, _, *s = input.shape
        assert simplex(input), f"Entropy input should be a simplex"
        e = input * (input + self._eps).log()
        e = -1.0 * e.sum(1)
        assert e.shape == torch.Size([b, *s])
        if self._reduction == "mean":
            return e.mean()
        elif self._reduction == "sum":
            return e.sum()
        else:
            return e


class Entropy_2D(Entropy):
    """
    Give Entropy 2D maps, `reduction` fixed to be none
    """

    def __init__(self, eps=1e-16):
        super().__init__("none", eps)

    def forward(self, input: torch.Tensor) -> Tensor:
        assert input.shape.__len__() == 4
        return super().forward(input)


class SimplexCrossEntropyLoss(nn.Module):
    def __init__(self, reduction="mean", eps=1e-16) -> None:
        super().__init__()
        _check_reduction_params(reduction)
        self._reduction = reduction
        self._eps = eps

    def forward(self, prob: Tensor, target: Tensor, **kwargs) -> Tensor:
        if not kwargs.get("disable_assert"):
            assert not target.requires_grad
            assert prob.requires_grad
            assert prob.shape == target.shape
            assert simplex(prob)
            assert simplex(target)
        b, c, *_ = target.shape
        ce_loss = (-target * torch.log(prob)).sum(1)
        if self._reduction == "mean":
            return ce_loss.mean()
        elif self._reduction == "sum":
            return ce_loss.sum()
        else:
            return ce_loss


class KL_div(nn.Module):
    r"""
    KL(p,q)= -\sum p(x) * log(q(x)/p(x))
    where p, q are distributions
    p is usually the fixed one like one hot coding
    p is the target and q is the distribution to get approached.

    reduction (string, optional): Specifies the reduction to apply to the output:
    ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
    ``'mean'``: the sum of the output will be divided by the number of
    elements in the output, ``'sum'``: the output will be summed.
    """

    def __init__(self, reduction="mean", eps=1e-16):
        super().__init__()
        _check_reduction_params(reduction)
        self._eps = eps
        self._reduction = reduction

    def forward(self, prob: Tensor, target: Tensor, **kwargs) -> Tensor:
        if not kwargs.get("disable_assert"):
            assert prob.shape == target.shape
            assert simplex(prob), prob
            assert simplex(target), target
            assert not target.requires_grad
            assert prob.requires_grad
        b, c, *_ = target.shape
        kl = (-target * torch.log((prob + self._eps) / (target + self._eps))).sum(1)
        if self._reduction == "mean":
            return kl.mean()
        elif self._reduction == "sum":
            return kl.sum()
        else:
            return kl


class KL_div_2D(KL_div):
    """
    Specific KL 2D interface
    """

    def __init__(self, eps=1e-16):
        super().__init__("none", eps)

    def forward(self, p_prob: Tensor, y_prob: Tensor, **kwargs):
        """
        :param p_probs:
        :param y_prob: the Y_logit is like that for cross-entropy
        :return: 2D map?
        """
        b, c, h, w = p_prob.shape
        kl_map = super().forward(p_prob, y_prob, **kwargs)
        assert kl_map.shape == torch.Size([b, h, w])
        return kl_map


class JSD_div(nn.Module):
    """
    general JS divergence interface
    :<math>{\rm JSD}_{\pi_1, \ldots, \pi_n}(P_1, P_2, \ldots, P_n) = H\left(\sum_{i=1}^n \pi_i P_i\right) - \sum_{i=1}^n \pi_i H(P_i)</math>


    reduction (string, optional): Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed.
    """

    def __init__(self, reduction="mean", eps=1e-16):
        super().__init__()
        _check_reduction_params(reduction)
        self._reduction = reduction
        self._eps = eps
        self._entropy_criterion = Entropy(reduction=reduction, eps=eps)

    def forward(self, *input: Tensor) -> Tensor:
        assert assert_list(
            lambda x: simplex(x), input
        ), f"input tensor should be a list of simplex."
        assert assert_list(
            lambda x: x.shape == input[0].shape, input
        ), "input tensor should have the same dimension"
        mean_prob = sum(input) / input.__len__()
        f_term = self._entropy_criterion(mean_prob)
        mean_entropy = sum(
            list(map(lambda x: self._entropy_criterion(x), input))
        ) / len(input)
        assert f_term.shape == mean_entropy.shape
        return f_term - mean_entropy


class JSD_div_2D(JSD_div):
    def __init__(self, eps=1e-16):
        super().__init__("none", eps)

    def forward(self, *input: Tensor) -> Tensor:
        b, c, h, w = input[0].shape
        jsd_map = super().forward(*input)
        assert jsd_map.shape == torch.Size([b, h, w])
        return jsd_map
