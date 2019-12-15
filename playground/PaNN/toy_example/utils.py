import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Subset


def convbnrelu_bloc(input_dim, output_dim):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=False,
        ),
        nn.BatchNorm2d(output_dim),
        nn.ReLU(inplace=True),
    )


class SimpleNet(nn.Module):
    def __init__(self, input_dim=1, num_classes=10) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.block1 = convbnrelu_bloc(input_dim, 16)
        self.downsample1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = convbnrelu_bloc(16, 32)
        self.downsample2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block3 = convbnrelu_bloc(32, 64)
        self.downsample3 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, input):
        output = self.downsample1(self.block1(input))
        output = self.downsample2(self.block2(output))
        output = self.downsample3(self.block3(output))
        output = self.fc(torch.flatten(output, 1))
        return F.softmax(output, 1)


def get_prior_from_dataset(dataset: Subset):
    import pandas as pd

    target = dataset.dataset.targets[dataset.indices]
    value_count = pd.Series(target).value_counts()
    return torch.from_numpy(
        (value_count.sort_index() / value_count.sum()).values
    ).float()


if __name__ == "__main__":
    input = torch.randn(10, 1, 28, 28)
    net = SimpleNet(1, num_classes=4)
    outoutp = net(input)
