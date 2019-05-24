import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..loss.loss import KL_div


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    # let the track_running_stats to be inverse
    model.apply(switch_attr)
    yield
    # let the track_running_stats to be inverse
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True)  # + 1e-8
    assert torch.allclose(d.view(d.shape[0], -1).norm(dim=1), torch.ones(d.shape[0]).to(d.device), rtol=1e-3)
    return d


_kl_div = KL_div(reduce=True)


# def _kl_div(log_probs, probs):
#     # pytorch KLDLoss is averaged over all dim if size_average=True
#     kld = F.kl_div(log_probs, probs, size_average=False)
#     return kld / log_probs.shape[0]


class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, prop_eps=0.25, ip=1):
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

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                adv_distance = _kl_div(F.softmax(pred_hat, dim=1), pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad.clone())
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps.view(-1, 1) * self.prop_eps
            pred_hat = model(x + r_adv)
            lds = _kl_div(F.softmax(pred_hat, dim=1), pred)

        return lds, (x + r_adv).detach(), r_adv.detach()
