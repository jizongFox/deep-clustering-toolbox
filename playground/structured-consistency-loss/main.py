from PIL import Image
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from deepclustering.utils import class2one_hot


def structured_consistency_loss(prediction: Tensor) -> Tensor:
    pass


if __name__ == "__main__":
    gt = np.array(Image.open(".data/patient001_01_0_6_gt.png"))
    gt_onehot = class2one_hot(torch.from_numpy(gt).float(), 4).float()
    gt_flatten = gt_onehot.view(1, 4, -1)
    distance = nn.PairwiseDistance()(gt_onehot, gt_onehot)
    print(distance)
