import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple
from ..loss import Entropy_2D


class FSGMGenerator(object):

    def __init__(self, net: nn.Module, eplision: float = 0.05) -> None:
        super().__init__()
        self.net = net
        self.eplision = eplision

    def __call__(self, img: Tensor, gt: Tensor, criterion: nn.Module) -> Tuple[Tensor, Tensor, Tensor]:
        assert img.shape.__len__() == 4
        assert img.shape[0] >= gt.shape[0]
        img.requires_grad = True
        if img.grad is not None:
            img.grad.zero_()
        self.net.zero_grad()
        pred = self.net(img)
        if img.shape[0] > gt.shape[0]:
            gt = torch.cat((gt, pred.max(1)[1][gt.shape[0]:].unsqueeze(1)), dim=0)
        loss = criterion(pred, gt.squeeze(1))
        loss.backward()
        adv_img, noise = self.adversarial_fgsm(img, img.grad, epsilon=self.eplision)
        self.net.zero_grad()
        img.grad.zero_()
        return adv_img.detach(), noise.detach(), F.softmax(pred, 1)

    @staticmethod
    # adversarial generation
    def adversarial_fgsm(image: Tensor, data_grad: Tensor, epsilon: float = 0.01) -> Tuple[Tensor, Tensor]:
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
        self.entropy = Entropy_2D()  # type: Tensor # shape:

    @staticmethod
    def _l2_normalize(d) -> Tensor:
        # d = d.cpu().numpy()
        # d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-16)
        # return torch.from_numpy(d)
        d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
        d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-16
        assert torch.allclose(d.view(d.shape[0], -1).norm(dim=1), torch.ones(d.shape[0]).to(d.device), rtol=1e-3)
        return d

    @staticmethod
    def kl_div_with_logit(q_logit, p_logit):
        '''
        :param q_logit:it is like the y in the ce loss
        :param p_logit: it is the logit to be proched to q_logit
        :return:
        '''
        q = F.softmax(q_logit, dim=1)
        logq = F.log_softmax(q_logit, dim=1)
        logp = F.log_softmax(p_logit, dim=1)

        qlogq = (q * logq).sum(dim=1)
        qlogp = (q * logp).sum(dim=1)
        return qlogq - qlogp

    def __call__(self, img: Tensor, loss_name='kl') -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            pred = self.net(img)
        # prepare random unit tensor
        d = torch.Tensor(img.size()).normal_()  # 所有元素的std =1, average = 0
        d = self._l2_normalize(d).to(img.device)
        assert torch.allclose(d.view(d.shape[0], -1).norm(dim=1),
                              torch.ones(d.shape[0]).to(img.device)), 'The L2 normalization fails'
        self.net.zero_grad()
        for _ in range(self.ip):
            d = self.xi * self._l2_normalize(d).to(img.device)
            d.requires_grad = True
            y_hat = self.net(img + d)

            delta_kl = self.kl_div_with_logit(pred.detach(), y_hat, self.axises)  # B/H/W
            delta_kl.mean().backward()

            d = d.grad.data.clone().cpu()
            self.net.zero_grad()
        ##
        d = self._l2_normalize(d).to(img.device)
        r_adv = self.eps * d
        # compute lds
        img_adv = img + r_adv.detach()
        img_adv = torch.clamp(img_adv, 0, 1)

        return img_adv.detach(), r_adv.detach()
