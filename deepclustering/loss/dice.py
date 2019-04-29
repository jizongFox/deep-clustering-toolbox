import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss

'''
A serie of dice loss taken from https://github.com/rogertrullo/pytorch/blob/rogertrullo-dice_loss/torch/nn/functional.py#L708
'''


def dice_coefficient(input, target, smooth=1.0):
    '''
    This is multiclass dice loss implementation.
    :param input:
    :param target:
    :param smooth:
    :return:
    '''

    """input : is a torch variable of size BatchxnclassesxHxW representing
    log probabilities for each class
    target : is a 1-hot representation of the groundtruth, shoud have same size
    as the input"""

    assert input.size() == target.size(), 'Input sizes must be equal.'
    assert input.dim() == 4, 'Input must be a 4D Tensor.'
    uniques = np.unique(target.data.cpu().numpy())
    assert set(list(uniques)) <= set([0, 1]), 'Target must only contain zeros and ones.'
    assert smooth > 0, 'Smooth must be greater than 0.'

    probs = F.softmax(input, dim=1)
    target_f = target.float()

    num = probs * target_f  # b, c, h, w -- p*g
    num = torch.sum(num, dim=3)  # b, c, h
    num = torch.sum(num, dim=2)  # b, c

    den1 = probs * probs  # b, c, h, w -- p^2
    den1 = torch.sum(den1, dim=3)  # b, c, h
    den1 = torch.sum(den1, dim=2)  # b, c

    den2 = target_f * target_f  # b, c, h, w -- g^2
    den2 = torch.sum(den2, dim=3)  # b, c, h
    den2 = torch.sum(den2, dim=2)  # b, c

    dice = (2 * num + smooth) / (den1 + den2 + smooth)

    return dice


def dice_loss(input, target, optimize_bg=False, weight=None,
              smooth=1.0, size_average=True, reduce=True):
    """input : is a torch variable of size BatchxnclassesxHxW representing
    log probabilities for each class
    target : is a 1-hot representation of the groundtruth, shoud have same size
    as the input

    weight (Variable, optional): a manual rescaling weight given to each
            class. If given, has to be a Variable of size "nclasses"""

    dice = dice_coefficient(input, target, smooth=smooth)

    if not optimize_bg:
        # we ignore bg dice val, and take the fg
        dice = dice[:, 1:]

    if not isinstance(weight, type(None)):
        if not optimize_bg:
            weight = weight[1:]  # ignore bg weight
        weight = weight.size(0) * weight / weight.sum()  # normalize fg weights
        dice = dice * weight  # weighting

    # loss is calculated using mean over fg dice vals
    dice_loss = 1 - dice.mean(1)

    if not reduce:
        return dice_loss

    if size_average:
        return dice_loss.mean()

    return dice_loss.sum()


class DiceLoss(_WeightedLoss):

    def __init__(self, optimize_bg=False, weight=None,
                 smooth=1.0, size_average=True, reduce=True):
        """input : is a torch variable of size BatchxnclassesxHxW representing
        log probabilities for each class
        target : is a 1-hot representation of the groundtruth, shoud have same
        size as the input

        weight (Variable, optional): a manual rescaling weight given to each
                class. If given, has to be a Variable of size "nclasses"""

        super(DiceLoss, self).__init__(weight, size_average)
        self.optimize_bg = optimize_bg
        self.smooth = smooth
        self.reduce = reduce
        self.size_average = True

    def forward(self, input, target):
        # _assert_no_grad(target)
        return dice_loss(input, target, optimize_bg=self.optimize_bg,
                         weight=self.weight, smooth=self.smooth,
                         size_average=self.size_average,
                         reduce=self.reduce)


class DiceCoefficient(torch.nn.Module):

    def __init__(self, smooth=1.0):
        """input : is a torch variable of size BatchxnclassesxHxW representing
        log probabilities for each class
        target : is a 1-hot representation of the groundtruth, shoud have same
        size as the input"""
        super(DiceCoefficient, self).__init__()

        self.smooth = smooth

    def forward(self, input, target):
        return dice_coefficient(input, target, smooth=self.smooth)
