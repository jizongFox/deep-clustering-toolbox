import torch.nn as nn

from .net6c import ClusterNet6c, ClusterNet6cTrunk
from .vgg import VGGNet
from deepclustering.decorator.decorator import export

__all__ = ["ClusterNet6cTwoHead", "ClusterNet6cTwoHead_Param"]


class ClusterNet6cTwoHeadHead(nn.Module):
    def __init__(
        self,
        input_size: int = 64,
        output_k: int = 10,
        num_sub_heads: int = 5,
        semisup=False,
        batchnorm_track: bool = True,
    ):
        r"""
        :param input_size: input_size of the image
        :param output_k: clustering number
        :param num_sub_heads: sub head number for one head
        :param semisup: whether use semi supervised.
        :param batchnorm_track: whether to track the batchnorm states.
        """
        super(ClusterNet6cTwoHeadHead, self).__init__()
        self.batchnorm_track = batchnorm_track
        self.cfg = ClusterNet6c.cfg
        num_features = self.cfg[-1][0]
        self.semisup = semisup
        if input_size in (24, 28):
            features_sp_size = 3
        elif input_size == 64:
            features_sp_size = 8
        elif input_size == 32:
            features_sp_size = 4
        else:
            raise ValueError(
                f"`input_size` should be in {24, 28, 32, 64}, given {input_size}."
            )
        if not semisup:
            self.num_sub_heads = num_sub_heads
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(
                            num_features * features_sp_size * features_sp_size, output_k
                        ),
                        nn.Softmax(dim=1),
                    )
                    for _ in range(self.num_sub_heads)
                ]
            )
        else:
            self.head = nn.Linear(
                num_features * features_sp_size * features_sp_size, output_k
            )

    def forward(self, x, kmeans_use_features=False):
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
class ClusterNet6cTwoHead(VGGNet):
    r"""
    Clustering based on VGG with two heads
    """
    cfg = [(64, 1), ("M", None), (128, 1), ("M", None), (256, 1), ("M", None), (512, 1)]

    def __init__(
        self,
        num_channel: int = 3,
        input_size: int = 64,
        output_k_A: int = 10,
        output_k_B: int = 10,
        num_sub_heads: int = 5,
        semisup: bool = False,
        batchnorm_track: bool = True,
    ):
        super(ClusterNet6cTwoHead, self).__init__()

        self.batchnorm_track = batchnorm_track

        self.trunk = ClusterNet6cTrunk(
            num_channel=num_channel, batchnorm_track=self.batchnorm_track
        )

        self.head_A = ClusterNet6cTwoHeadHead(
            input_size=input_size,
            output_k=output_k_A,
            num_sub_heads=num_sub_heads,
            semisup=semisup,
            batchnorm_track=self.batchnorm_track,
        )

        self.head_B = ClusterNet6cTwoHeadHead(
            input_size=input_size,
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
        if penultimate_features:
            print("Not needed/implemented for this arch")
            exit(1)
        # default is "B" for use by eval IIC
        # training script switches between A and B
        x = self.trunk(x)
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


ClusterNet6cTwoHead_Param = {
    "num_channel": 3,
    "input_size": 64,
    "output_k_A": 10,
    "output_k_B": 10,
    "num_sub_heads": 5,
    "semisup": False,
}
