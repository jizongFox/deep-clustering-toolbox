import torch
from torch import nn, Tensor


class MetaDice(object):
    """
    3D and 2D dice computator
    """

    def __init__(self, method: str, weight: Tensor = None, reduce: bool = False, eps: float = 1e-8) -> None:
        """
        :param method must be in (2d, 3d)
        :param weight: Weight to be multipled to each class.
        :param eps: default to 1e-8
        :param reduce: if reduce classwise mean. mean on batch samples.
        :return:
        """
        assert method in ('2d', '3d'), method
        self.method = method
        self.reduce = reduce
        self.eps = eps
        self.weight = weight

    def __call__(self, pred: Tensor, target: Tensor):
        """
        :param pred: softmax or one_hot prediction, with or without gradient. having the shape of B C H W etc
        :param target: One_hot mask of the target, must have the same shape as the `pred`
        :param pred:
        :param target:
        :return:
        """
        assert pred.shape == target.shape, f"`pred` and `target` should have the same shape, " \
            f"given `pred`:{pred.shape}, `target`:{target.shape}."
        assert not target.requires_grad
        pred, target = pred.float(), target.float()

        B, C, *hw = pred.shape
        reduce_axises = list(range(2, pred.shape.__len__())) if self.method == '2d' else \
            [0] + list(range(2, pred.shape.__len__()))
        intersect = (pred * target).sum(reduce_axises)
        union = (pred + target).sum(reduce_axises)

        # TODO: add the weight here.
        if self.weight is not None:
            intersect = self.weight * intersect

        dices = 2. * intersect / union.clamp(min=self.eps)
        assert dices.shape == torch.Size([B, C]) if self.method == '2d' else torch.Size([C])
        if self.reduce and self.method == '2d':
            return dices.mean(0)
        return dices


dice_coef = MetaDice(method='2d', reduce=False)
dice_batch = MetaDice(method='3d', reduce=False)  # used for 3d dice


class DiceLoss(nn.Module):
    """Computes Dice Loss, which just 1 - DiceCoefficient described above.
    Additionally allows per-class weights to be provided.
    """

    def __init__(self, weight=None, ignore_index=None, sigmoid_normalization=False,
                 skip_last_target=False, epsilon=1e-5, ):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify sigmoid_normalization=False.
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)
        # if True skip the last channel in the target
        self.skip_last_target = skip_last_target

    def forward(self, input, target):
        # get probabilities from logits
        # input = self.normalization(input)
        if self.weight is not None:
            weight = self.weight
        else:
            weight = None

        if self.skip_last_target:
            target = target[:, :-1, ...]

        per_channel_dice = compute_per_channel_dice(input, target, epsilon=self.epsilon, ignore_index=self.ignore_index,
                                                    weight=weight)
        # Average the Dice score across all channels/classes
        return 1. - per_channel_dice


class GeneralizedDiceLoss(nn.Module):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)  # if you want to store it in the state_dict but not in the parameters()
        self.ignore_index = ignore_index
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        assert input.size() == target.size(), "'input' and 'target' must have the same shape"
        # so the target here is the onehot

        # mask ignore_index if present
        if self.ignore_index is not None:
            mask = target.clone().ne_(self.ignore_index)
            mask.requires_grad = False

            input = input * mask
            target = target * mask

        input = flatten(input)
        target = flatten(target)

        target = target.float()
        target_sum = target.sum(-1)
        class_weights = 1. / (target_sum * target_sum).clamp(min=self.epsilon)

        intersect = (input * target).sum(-1) * class_weights
        if self.weight is not None:
            weight = self.weight
            intersect = weight * intersect
        intersect = intersect.sum()

        denominator = ((input + target).sum(-1) * class_weights).sum()

        return 1. - 2. * intersect / denominator.clamp(min=self.epsilon)


def compute_per_channel_dice(input, target, epsilon=1e-5, ignore_index=None, weight=None):
    # assumes that input is a normalized probability

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # mask ignore_index if present
    if ignore_index is not None:
        mask = target.clone().ne_(ignore_index)
        mask.requires_grad = False

        input = input * mask
        target = target * mask

    input = flatten(input)
    target = flatten(target)

    target = target.float()
    # Compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    denominator = (input + target).sum(-1)
    return 2. * intersect / denominator.clamp(min=epsilon)


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order).contiguous()
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)
