import warnings
from typing import List

import torch
import torch.nn as nn

from .net5g import ClusterNet5gTrunk
from .residual import BasicBlock, ResNet
from deepclustering.decorator.decorator import export

# resnet34 and full channels

__all__ = ["ClusterNet5gMultiHead", "ClusterNet5gMultiHead_Param"]


class ClusterNet5gMultiHeadHead(nn.Module):
    def __init__(
        self,
        output_k: int,
        num_sub_heads: int,
        semisup: bool = False,
        batchnorm_track: bool = True,
    ):
        super(ClusterNet5gMultiHeadHead, self).__init__()
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
class ClusterNet5gMultiHead(ResNet):
    """
    based on resnet with two heads (multiple subheads). One head is for overclustering and
    other is for normal clustering
    """

    num_name_mapping = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G"}
    name_num_mapping = {v: k for k, v in num_name_mapping.items()}

    def __init__(
        self,
        num_channel: int = 3,
        output_k_list: List[int] = [70, 10],
        semisup: bool = False,
        num_sub_heads: int = 5,
        batchnorm_track: bool = True,
        verbose=False,
    ):
        r"""
        :param input_size: image size of the raw image, only support 96, 64, 32
        :param num_channel: image channel
        :param output_k_list: list of clustering nums for clusterings. last one should be the ground truth class_num
        :param semisup: return semi supervised feature
        :param num_sub_heads: sub-head number to form an ensemble-like prediction for each head
        :param batchnorm_track:  whether to track the batchnorm states
        """
        super(ClusterNet5gMultiHead, self).__init__()
        if isinstance(output_k_list, int):
            output_k_list = [output_k_list]
        assert isinstance(
            output_k_list, (list, tuple)
        ), f"output_k_list should be a list or tuple, given {output_k_list}."

        self.output_k_list: List[int] = output_k_list
        self.batchnorm_track = batchnorm_track
        # resnet structure
        self.trunk = ClusterNet5gTrunk(
            num_channel=num_channel, batchnorm_track=self.batchnorm_track
        )
        for head_i, cluster_num in enumerate(self.output_k_list):
            setattr(
                self,
                f"head_{self.num_name_mapping[head_i + 1]}",
                ClusterNet5gMultiHeadHead(
                    output_k=cluster_num,
                    num_sub_heads=num_sub_heads,
                    semisup=semisup,
                    batchnorm_track=self.batchnorm_track,
                ),
            )
        self.verbose = verbose
        if self.verbose:
            print("semisup: %s" % semisup)
        self._initialize_weights()

    def forward(
        self,
        x,
        head=None,
        kmeans_use_features=False,
        trunk_features=False,
        penultimate_features=False,
    ):
        if head is None:
            warnings.warn(
                "head is None, using the last head: head_%s."
                % self.num_name_mapping[len(self.output_k_list)]
            )
            head = self.num_name_mapping[len(self.output_k_list)]

        assert isinstance(head, str) and head in list(self.name_num_mapping.keys()), (
            f"head given {head} should be "
            f"within {', '.join(list(self.name_num_mapping.keys())[:len(self.output_k_list)])}."
        )
        # default is "B" for use by eval IIC
        # training script switches between A and B
        x = self.trunk(x, penultimate_features=penultimate_features)
        if trunk_features:  # for semisup
            return x
        else:
            x = getattr(self, f"head_{head}")(
                x, kmeans_use_features=kmeans_use_features
            )
            return x


ClusterNet5gMultiHead_Param = {
    "num_channel": 3,
    "output_k_list": [150, 70, 10],
    "num_sub_heads": 5,
    "semisup": False,
}
