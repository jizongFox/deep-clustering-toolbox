import warnings

import torch.nn as nn
from deepclustering.arch.classification.IIC.residual import (
    BasicBlock,
    ResNet,
    ResNetTrunk,
)

# resnet34 and full channels

__all__ = ["ClusterNet5g", "ClusterNet5g_Param"]


class ClusterNet5gTrunk(ResNetTrunk):
    r"""
    ResNet based Trunk model
    """

    def __init__(
        self, input_size: int, num_channel: int = 3, batchnorm_track: bool = True
    ):
        """
        ResNet Trunk Initialization
        :param input_size: the input image size
        :param num_channel: image channel, 3 for RGB while 1 for grey image
        :param batchnorm_track: if track the batchnorm state
        """
        super(ClusterNet5gTrunk, self).__init__()

        self.batchnorm_track = batchnorm_track

        block = BasicBlock
        layers = [3, 4, 6, 3]

        in_channels = num_channel
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=self.batchnorm_track)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if input_size == 96:
            avg_pool_sz = 7
        elif input_size == 64:
            avg_pool_sz = 5
        elif input_size == 32:
            avg_pool_sz = 3
        else:
            raise ValueError(
                f"the input size should be in (96, 64, 32), given {input_size}"
            )
        print("avg_pool_sz %d" % avg_pool_sz)

        self.avgpool = nn.AvgPool2d(avg_pool_sz, stride=1)

    def forward(self, x, penultimate_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if not penultimate_features:
            x = self.layer4(x)
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


class ClusterNet5gHead(nn.Module):
    def __init__(
        self, output_k: int, num_sub_heads: int, batchnorm_track: bool = True
    ) -> None:
        r"""
        :param output_k: number of clustering
        :param num_sub_heads: number of sub heads to form an ensemble-like prediction
        :param batchnorm_track: track the batchnorm
        """
        super(ClusterNet5gHead, self).__init__()
        self.batchnorm_track = batchnorm_track
        self.num_sub_heads = num_sub_heads
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(512 * BasicBlock.expansion, output_k), nn.Softmax(dim=1)
                )
                for _ in range(self.num_sub_heads)
            ]
        )

    def forward(self, x, kmeans_use_features=False):
        r"""
        :param x: feature from trunk
        :param kmeans_use_features: whether use kmeans_use_features
        :return: predictions with sub_heads
        """
        results = []
        for i in range(self.num_sub_heads):
            if kmeans_use_features:
                results.append(x)  # duplicates
            else:
                results.append(self.heads[i](x))
        return results


class ClusterNet5g(ResNet):
    r"""
        Clustering model based on ResNet with ResNetTrunk and ResNetHead
    """

    def __init__(
        self,
        input_size: int,
        num_channel: int = 3,
        output_k: int = 10,
        num_sub_heads: int = 5,
        batchnorm_track: bool = True,
        *args,
        **kwargs,
    ):
        r"""
        :param input_size: image size of the raw image, only support 96, 64, 32
        :param num_channel: image channel
        :param output_k: clustering number
        :param num_sub_heads: sub-head number to form an ensemble-like prediction
        :param batchnorm_track: whether to track the batchnorm states
        """
        if len(args) > 0:
            warnings.warn(f"Received unassigned args with args: {args}.")
        if len(kwargs) > 0:
            kwarg_str = ", ".join([f"{k}:{v}" for k, v in kwargs.items()])
            warnings.warn(f"Received unassigned kwargs: \n{kwarg_str}")
        super(ClusterNet5g, self).__init__()

        self.batchnorm_track = batchnorm_track

        self.trunk = ClusterNet5gTrunk(
            input_size=input_size,
            num_channel=num_channel,
            batchnorm_track=self.batchnorm_track,
        )
        self.head = ClusterNet5gHead(
            output_k=output_k,
            num_sub_heads=num_sub_heads,
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
        x = self.trunk(x, penultimate_features=penultimate_features)

        if trunk_features:  # for semisup
            return x

        preds = self.head(x, kmeans_use_features=kmeans_use_features)  # returns list
        return preds, x


ClusterNet5g_Param = {
    "input_size": 64,
    "num_channel": 3,
    "output_k": 10,
    "num_sub_heads": 5,
}
if __name__ == "__main__":
    import torch

    net = ClusterNet5g(input_size=32, num_channel=1, output_k=10, num_sub_heads=1)
    img = torch.randn(8, 1, 32, 32)
    p, z = net(img)
    print()
