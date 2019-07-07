from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from deepclustering.decorator.decorator import _disable_tracking_bn_stats


class FSGMGenerator(object):
    def __init__(self, net: nn.Module, eplision: float = 0.05) -> None:
        super().__init__()
        self.net = net
        self.eplision = eplision

    def __call__(
        self, img: Tensor, gt: Tensor, criterion: nn.Module
    ) -> Tuple[Tensor, Tensor, Tensor]:
        assert img.shape.__len__() == 4
        assert img.shape[0] >= gt.shape[0]
        img.requires_grad = True
        if img.grad is not None:
            img.grad.zero_()
        self.net.zero_grad()
        pred = self.net(img)
        if img.shape[0] > gt.shape[0]:
            gt = torch.cat((gt, pred.max(1)[1][gt.shape[0] :].unsqueeze(1)), dim=0)
        loss = criterion(pred, gt.squeeze(1))
        loss.backward()
        adv_img, noise = self.adversarial_fgsm(img, img.grad, epsilon=self.eplision)
        self.net.zero_grad()
        img.grad.zero_()
        return adv_img.detach(), noise.detach(), F.softmax(pred, 1)

    @staticmethod
    # adversarial generation
    def adversarial_fgsm(
        image: Tensor, data_grad: Tensor, epsilon: float = 0.01
    ) -> Tuple[Tensor, Tensor]:
        """
        FGSM for generating adversarial sample
        :param image: original clean image
        :param epsilon: the pixel-wise perturbation amount
        :param data_grad: gradient of the loss w.r.t the input image
        :return: perturbed image representing adversarial sample
        """
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        noise = epsilon * sign_data_grad
        perturbed_image = image + noise
        # Adding clipping to maintain [0,1] range
        # perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image.detach(), noise.detach()


class VATGenerator(object):
    def __init__(self, net: nn.Module, xi=1e-6, eplision=10, ip=1) -> None:
        """VAT generator based on https://arxiv.org/abs/1704.03976
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATGenerator, self).__init__()
        self.xi = xi
        self.eps = eplision
        self.ip = ip
        self.net = net

    @staticmethod
    def _l2_normalize(d: Tensor) -> Tensor:
        d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
        d /= torch.norm(d_reshaped, dim=1, keepdim=True)
        assert torch.allclose(
            d.view(d.shape[0], -1).norm(dim=1),
            torch.ones(d.shape[0]).to(d.device),
            rtol=1e-3,
        )
        return d

    @staticmethod
    def kl_div_with_logit(q_logit: Tensor, p_logit: Tensor):
        """
        :param q_logit:it is like the y in the ce loss
        :param p_logit: it is the logit to be proched to q_logit
        :return:
        """
        assert (
            not q_logit.requires_grad
        ), f"q_logit should be no differentiable, like y."
        assert p_logit.requires_grad
        q = F.softmax(q_logit, dim=1)
        logq = F.log_softmax(q_logit, dim=1)
        logp = F.log_softmax(p_logit, dim=1)

        qlogq = (q * logq).sum(dim=1)
        qlogp = (q * logp).sum(dim=1)
        return qlogq - qlogp

    def __call__(self, img: Tensor, loss_name="kl") -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            pred = self.net(img)
        # prepare random unit tensor
        d = torch.Tensor(img.size()).normal_()  # 所有元素的std =1, average = 0
        d = self._l2_normalize(d).to(img.device)
        self.net.zero_grad()
        with _disable_tracking_bn_stats(self.net):
            for _ in range(self.ip):
                d = self.xi * self._l2_normalize(d)
                d.requires_grad = True
                y_hat = self.net(img + d)

                delta_kl = self.kl_div_with_logit(pred.detach(), y_hat)  # B/H/W
                delta_kl.mean().backward()

                d = d.grad.data.clone()
                self.net.zero_grad()
            ##
            d = self._l2_normalize(d)
            r_adv = 0.25 * self.eps.view(-1, 1) * d
            # compute lds
            img_adv = img + r_adv.detach()
            # img_adv = torch.clamp(img_adv, 0, 1)

        return img_adv.detach(), r_adv.detach()


def vat(network, x, eps_list, xi=10, Ip=1):
    # compute the regularized penality [eq. (4) & eq. (6), 1]

    with torch.no_grad():
        y = network(x)
        d = torch.randn((x.size()[0], x.size()[1]))
    d = F.normalize(d, p=2, dim=1)
    for ip in range(Ip):
        d_var = d

        d_var = d_var.to(x.device)
        d_var.requires_grad_(True)
        y_p = network(x + xi * d_var, update_batch_stats=False)
        kl_loss = distance(y, y_p)
        kl_loss.backward()
        d = d_var.grad
        d = F.normalize(d, p=2, dim=1)
    d_var = d

    d_var = d_var.to(x.device)
    eps = 0.25 * eps_list
    eps = eps.view(-1, 1)
    # y_2 = network(x + eps * d_var)
    # return distance(y, y_2)
    return (x + eps * d_var).detach(), (eps * d_var).detach()


def distance(y0, y1):
    # compute KL divergence between the outputs of the newtrok
    return kl(F.softmax(y0, dim=1), F.softmax(y1, dim=1))


def kl(p, q):
    # compute KL divergence between p and q
    return torch.sum(p * torch.log((p + 1e-8) / (q + 1e-8))) / float(len(p))
