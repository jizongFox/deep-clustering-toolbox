import torch.nn as nn

from .vgg import VGGTrunk, VGGNet
from deepclustering.decorator.decorator import export

# 4h but for cifar, 24x24

__all__ = ["ClusterNet6c", "ClusterNet6c_Param"]


class ClusterNet6cTrunk(VGGTrunk):
    def __init__(self, num_channel: int = 3, batchnorm_track: bool = True):
        r"""
        Initialize
        :param num_channel: input image channel, default 3 
        :param batchnorm_track:
        """
        super(ClusterNet6cTrunk, self).__init__()

        self.batchnorm_track = batchnorm_track
        self.conv_size = 5
        self.pad = 2
        self.cfg = ClusterNet6c.cfg
        self.in_channels = num_channel
        self.features = self._make_layers()

    def forward(self, x):
        x = self.features(x)
        bn, nf, h, w = x.size()
        x = x.view(bn, nf * h * w)
        return x


class ClusterNet6cHead(nn.Module):
    def __init__(
        self,
        input_size: int = 64,
        num_sub_heads: int = 5,
        output_k: int = 10,
        batchnorm_track: bool = True,
    ):
        super(ClusterNet6cHead, self).__init__()
        self.batchnorm_track = batchnorm_track
        self.num_sub_heads = num_sub_heads
        self.cfg = ClusterNet6c.cfg
        num_features = self.cfg[-1][0]
        if input_size == 24:
            features_sp_size = 3
        elif input_size == 64:
            features_sp_size = 8
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

    def forward(self, x, kmeans_use_features=False):
        results = []
        for i in range(self.num_sub_heads):
            if kmeans_use_features:
                results.append(x)  # duplicates
            else:
                results.append(self.heads[i](x))
        return results


@export
class ClusterNet6c(VGGNet):
    r"""
    VGG based clustering method with single head
    """
    cfg = [(64, 1), ("M", None), (128, 1), ("M", None), (256, 1), ("M", None), (512, 1)]

    def __init__(
        self,
        num_channel: int = 3,
        input_size: int = 64,
        num_sub_heads: int = 5,
        output_k: int = 10,
        batchnorm_track: bool = True,
    ):
        r"""
        :param num_channel: input image channel
        :param input_size: input image size
        :param num_sub_heads: num of sub heads for one head
        :param output_k: clustering numbers
        :param batchnorm_track: whether to track the batchnorm states
        """
        super(ClusterNet6c, self).__init__()
        self.batchnorm_track = batchnorm_track
        self.trunk = ClusterNet6cTrunk(
            num_channel=num_channel, batchnorm_track=self.batchnorm_track
        )
        self.head = ClusterNet6cHead(
            input_size=input_size,
            num_sub_heads=num_sub_heads,
            output_k=output_k,
            batchnorm_track=self.batchnorm_track,
        )

        self._initialize_weights()

    def forward(
        self,
        x,
        kmeans_use_features=False,
        trunk_features=False,
        penultimate_features=False,
    ):
        if penultimate_features:
            print("Not needed/implemented for this arch")
            exit(1)
        x = self.trunk(x)
        if trunk_features:  # for semisup
            return x
        x = self.head(x, kmeans_use_features=kmeans_use_features)  # returns list
        return x


ClusterNet6c_Param = {
    "num_channel": 3,
    "input_size": 64,
    "num_sub_heads": 5,
    "output_k": 10,
}
