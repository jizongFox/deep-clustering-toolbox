import torch
import torch.nn as nn

from .net5g import ClusterNet5gTrunk
from .residual import BasicBlock, ResNet
from deepclustering.decorator.decorator import export

# resnet34 and full channels

__all__ = ["ClusterNet5gTwoHead", "ClusterNet5gTwoHead_Param"]


class ClusterNet5gTwoHeadHead(nn.Module):
    def __init__(
        self,
        output_k: int,
        num_sub_heads: int,
        semisup: bool = False,
        batchnorm_track: bool = True,
    ):
        super(ClusterNet5gTwoHeadHead, self).__init__()
        self.batchnorm_track = batchnorm_track
        self.semisup = semisup
        if not self.semisup:
            """ Here the difference between semisup and not are the Softmax layer."""
            self.num_sub_heads = num_sub_heads
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(512 * BasicBlock.expansion, output_k),
                        nn.Softmax(dim=1),
                    )
                    for _ in range(self.num_sub_heads)
                ]
            )
        else:
            self.head = nn.Linear(512 * BasicBlock.expansion, output_k)

    def forward(self, x: torch.Tensor, kmeans_use_features: bool = False):
        if not self.semisup:
            results = []
            for i in range(self.num_sub_heads):
                if kmeans_use_features:
                    results.append(x)  # duplicates
                else:
                    results.append(self.heads[i](x))
            return results
        else:
            return self.head(x)


@export
class ClusterNet5gTwoHead(ResNet):
    """
    based on resnet with two heads (multiple subheads). One head is for overclustering and
    other is for normal clustering
    """

    def __init__(
        self,
        num_channel: int = 3,
        output_k_A: int = 10,
        output_k_B: int = 10,
        semisup: bool = False,
        num_sub_heads: int = 5,
        batchnorm_track: bool = True,
        verbose=False,
    ):
        r"""
        :param input_size: image size of the raw image, only support 96, 64, 32
        :param num_channel: image channel
        :param output_k_A: clustering num for over clustering
        :param output_k_B: clustering num for normal clustering
        :param semisup: return semi supervised feature
        :param num_sub_heads: sub-head number to form an ensemble-like prediction for each head
        :param batchnorm_track:  whether to track the batchnorm states
        """
        super(ClusterNet5gTwoHead, self).__init__()

        self.batchnorm_track = batchnorm_track
        # resnet structure
        self.trunk = ClusterNet5gTrunk(
            num_channel=num_channel, batchnorm_track=self.batchnorm_track
        )
        self.head_A = ClusterNet5gTwoHeadHead(
            output_k=output_k_A,
            num_sub_heads=num_sub_heads,
            semisup=semisup,
            batchnorm_track=self.batchnorm_track,
        )
        self.verbose = verbose
        if self.verbose:
            print("semisup: %s" % semisup)
        self.head_B = ClusterNet5gTwoHeadHead(
            output_k=output_k_B,
            num_sub_heads=num_sub_heads,
            semisup=semisup,
            batchnorm_track=self.batchnorm_track,
        )
        self._initialize_weights()

    def forward(
        self,
        x,
        head="B",
        kmeans_use_features=False,
        trunk_features=False,
        penultimate_features=False,
    ):
        # default is "B" for use by eval IIC
        # training script switches between A and B
        x = self.trunk(x, penultimate_features=penultimate_features)
        if trunk_features:  # for semisup
            return x
        # returns list or single
        if head == "A":
            x = self.head_A(x, kmeans_use_features=kmeans_use_features)
        elif head == "B":
            x = self.head_B(x, kmeans_use_features=kmeans_use_features)
        else:
            assert False
        return x


ClusterNet5gTwoHead_Param = {
    "num_channel": 3,
    "output_k_A": 70,
    "output_k_B": 10,
    "num_sub_heads": 5,
    "semisup": False,
}
