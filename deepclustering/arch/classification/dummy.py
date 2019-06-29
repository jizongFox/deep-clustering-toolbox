from torch import nn


class PlaceholderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(128, 256, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        return input
        # return self.avg_pool(self.conv1(input))


class Dummy(nn.Module):
    """
    This is a dummy network for debug
    """

    def __init__(self, num_channel=3, output_k=10, num_sub_heads=2):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(num_channel, 100, 3, padding=1),
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stride=2, kernel_size=3),
            nn.Conv2d(100, 50, 3, 1, 1),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stride=2, kernel_size=3),
            nn.Conv2d(50, 10, 3, 1, 1),
            nn.BatchNorm2d(10),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.num_sub_heads = num_sub_heads
        self.classifiers = nn.ModuleList()
        for i in range(num_sub_heads):
            self.classifiers.append(
                nn.Sequential(nn.Linear(10, output_k), nn.Softmax(1))
            )

    def forward(self, input):
        feature = self.feature(input)
        feature = feature.view(feature.size(0), -1)
        preds = []
        for i in range(self.num_sub_heads):
            _pred = self.classifiers[i](feature)
            preds.append(_pred)

        return preds


Dummy_Param = {"num_channel": 3, "output_k": 10, "num_sub_heads": 5}
