import contextlib
from typing import Tuple

import torch
import torch.nn as nn

from ..loss.loss import KL_div
from ..model import Model
from ..utils import simplex, assert_list


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, "track_running_stats"):
            m.track_running_stats ^= True

    # let the track_running_stats to be inverse
    model.apply(switch_attr)
    # return the model
    yield
    # let the track_running_stats to be inverse
    model.apply(switch_attr)


def _l2_normalize(d: torch.Tensor) -> torch.Tensor:
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True)  # + 1e-8
    ones_ = torch.ones(d.shape[0], device=d.device)
    assert torch.allclose(d.view(d.shape[0], -1).norm(dim=1), ones_, rtol=1e-3)
    return d


class VATLoss(nn.Module):
    def __init__(
        self, xi=10.0, eps=1.0, prop_eps=0.25, ip=1, distance_func=KL_div(reduce=True)
    ):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.prop_eps = prop_eps
        self.distance_func = distance_func

    def forward(self, model, x: torch.Tensor):
        """
        We support the output of the model would be a simplex.
        :param model:
        :param x:
        :return:
        """
        with torch.no_grad():
            pred = model(x)[0]
        assert simplex(pred)

        # prepare random unit tensor
        d = torch.randn_like(x, device=x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)[0]
                adv_distance = self.distance_func(pred_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)

            # calc LDS
            if isinstance(self.eps, torch.Tensor):
                # a dictionary is given
                bn, *shape = x.shape
                basic_view_shape: Tuple[int, ...] = (bn, *([1] * len(shape)))
                r_adv = d * self.eps.view(basic_view_shape).expand_as(d) * self.prop_eps
            elif isinstance(self.eps, (float, int)):
                r_adv = d * self.eps * self.prop_eps
            else:
                raise NotImplementedError(
                    f"eps should be tensor or float, given {self.eps}."
                )

            pred_hat = model(x + r_adv)[0]
            lds = self.distance_func(pred_hat, pred)

        return lds, (x + r_adv).detach(), r_adv.detach()


class VATLoss_Multihead(nn.Module):
    """
    this is the VAT for the multihead networks. each head outputs a simplex.
    """

    def __init__(
        self, xi=10.0, eps=1.0, prop_eps=0.25, ip=1, distance_func=KL_div(reduce=True)
    ):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss_Multihead, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.prop_eps = prop_eps
        self.distance_func = distance_func

    def forward(self, model: Model, x: torch.Tensor):
        with torch.no_grad():
            pred = model(x)
        assert assert_list(simplex, pred), f"pred should be a list of simplex."

        # prepare random unit tensor
        d = torch.randn_like(x, device=x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                assert assert_list(simplex, pred_hat)
                # here the pred_hat is the list of simplex
                adv_distance = list(
                    map(lambda p_, p: self.distance_func(p_, p), pred_hat, pred)
                )
                _adv_distance: torch.Tensor = sum(adv_distance) / float(
                    len(adv_distance)
                )
                _adv_distance.backward()  # type: ignore
                d = _l2_normalize(d.grad)

            # calc LDS
            if isinstance(self.eps, torch.Tensor):
                # a dictionary is given
                bn, *shape = x.shape
                basic_view_shape: Tuple[int, ...] = (bn, *([1] * len(shape)))
                r_adv = d * self.eps.view(basic_view_shape).expand_as(d) * self.prop_eps
            elif isinstance(self.eps, (float, int)):
                r_adv = d * self.eps * self.prop_eps
            else:
                raise NotImplementedError(
                    f"eps should be tensor or float, given {self.eps}."
                )

            pred_hat = model(x + r_adv)
            assert assert_list(simplex, pred_hat)
            lds = list(
                map(lambda p_, p: self.distance_func(p_, p), pred_hat, pred)
            )  # type: ignore
            lds: torch.Tensor = sum(lds) / float(len(lds))

        return lds, (x + r_adv).detach(), r_adv.detach()
