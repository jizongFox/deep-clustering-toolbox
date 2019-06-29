"""
This is taken from the IIC paper.
"""
import sys

import torch
from torch import Tensor
from torch import nn

from ..utils.general import simplex


class IIDLoss(nn.Module):
    def __init__(self, lamb: float = 1.0, eps: float = sys.float_info.epsilon):
        """
        :param lamb:
        :param eps:
        """
        super().__init__()
        self.lamb = float(lamb)
        self.eps = float(eps)

    def forward(self, x_out: Tensor, x_tf_out: Tensor):
        """
        return the inverse of the MI. if the x_out == y_out, return the inverse of Entropy
        :param x_out:
        :param x_tf_out:
        :return:
        """
        assert simplex(x_out), f"x_out not normalized."
        assert simplex(x_tf_out), f"x_tf_out not normalized."
        _, k = x_out.size()
        p_i_j = compute_joint(x_out, x_tf_out)
        assert p_i_j.size() == (k, k)

        p_i = (
            p_i_j.sum(dim=1).view(k, 1).expand(k, k)
        )  # p_i should be the mean of the x_out
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  # but should be same, symmetric

        # p_i = x_out.mean(0).view(k, 1).expand(k, k)
        # p_j = x_tf_out.mean(0).view(1, k).expand(k, k)

        # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
        p_i_j[p_i_j < self.eps] = self.eps
        p_j[p_j < self.eps] = self.eps
        p_i[p_i < self.eps] = self.eps

        loss = -p_i_j * (
            torch.log(p_i_j) - self.lamb * torch.log(p_j) - self.lamb * torch.log(p_i)
        )
        loss = loss.sum()
        loss_no_lamb = -p_i_j * (torch.log(p_i_j) - torch.log(p_j) - torch.log(p_i))
        loss_no_lamb = loss_no_lamb.sum()
        return loss, loss_no_lamb


def compute_joint(x_out: Tensor, x_tf_out: Tensor) -> Tensor:
    r"""
    return joint probability
    :param x_out: p1, simplex
    :param x_tf_out: p2, simplex
    :return: joint probability
    """
    # produces variable that requires grad (since args require grad)
    assert simplex(x_out), f"x_out not normalized."
    assert simplex(x_tf_out), f"x_tf_out not normalized."

    bn, k = x_out.shape
    assert x_tf_out.size(0) == bn and x_tf_out.size(1) == k

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k aggregated over one batch
    p_i_j = (p_i_j + p_i_j.t()) / 2.0  # symmetric
    p_i_j /= p_i_j.sum()  # normalise

    return p_i_j
