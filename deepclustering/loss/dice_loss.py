import torch
from torch import nn, Tensor

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


class TwoDimDiceLoss(MetaDice):
    """Computes Dice Loss, which just 1 - DiceCoefficient described above.
    Additionally allows per-class weights to be provided.
    return the classwise loss average on individual images.
    # todo: not sure how to handle with the ignore index
    """

    def __init__(
        self, weight: Tensor = None, ignore_index: int = None, reduce: bool = False
    ) -> None:
        super(TwoDimDiceLoss, self).__init__("2d", weight, reduce, 1e-8)
        self.ignore_index = ignore_index

    def forward(self, pred: Tensor, target: Tensor):
        assert simplex(pred)
        assert one_hot(target)
        if self.ignore_index is not None:
            mask = target.clone().ne_(self.ignore_index).type(torch.float)
            mask.requires_grad = False
            pred = pred * mask
            target = target.float() * mask

        dices = super().forward(pred, target)
        loss = (1.0 - dices).mean()

        return loss, dices


class ThreeDimDiceLoss(MetaDice):
    def __init__(self, weight: Tensor = None, ignore_index: int = None) -> None:
        super().__init__("3d", weight, True, 1e-8)
        self.ignore_index = ignore_index

    def forward(self, pred: Tensor, target: Tensor):
        assert simplex(pred)
        assert one_hot(target)
        if self.ignore_index is not None:
            mask = target.clone().ne_(self.ignore_index).type(torch.float)
            mask.requires_grad = False
            pred = pred * mask
            target = target.float() * mask

        dices = super().forward(pred, target)
        loss = (1.0 - dices).mean()

        return loss, dices


class GeneralizedDiceLoss(nn.Module):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(
        self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True
    ):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer(
            "weight", weight
        )  # if you want to store it in the state_dict but not in the parameters()
        self.ignore_index = ignore_index
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        assert (
            input.size() == target.size()
        ), "'input' and 'target' must have the same shape"
        # so the target here is the onehot

        # mask ignore_index if present
        if self.ignore_index is not None:
            mask = target.clone().ne_(self.ignore_index)
            mask.requires_grad = False

            input = input * mask
            target = target * mask

        # input = flatten(input)
        # target = flatten(target)

        target = target.float()
        target_sum = target.sum(-1)
        class_weights = 1.0 / (target_sum * target_sum).clamp(min=self.epsilon)

        intersect = (input * target).sum(-1) * class_weights
        if self.weight is not None:
            weight = self.weight
            intersect = weight * intersect
        intersect = intersect.sum()

        denominator = ((input + target).sum(-1) * class_weights).sum()

        return 1.0 - 2.0 * intersect / denominator.clamp(min=self.epsilon)
