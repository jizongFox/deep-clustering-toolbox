from apex import amp
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torchvision.transforms import ToTensor
import numpy as np
from deepclustering.meters import ConfusionMatrix
from deepclustering.utils import tqdm_
from deepclustering.model import Model, to_Apex, ZeroGradientBackwardStep


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 10)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        out = F.relu(self.fc1(input), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = self.fc3(out)
        return out


net = Net()
optimizer = optim.Adam(net.parameters(), lr=1e-5)
model = Model()
model.torchnet = net
model.optimizer = optimizer
model = to_Apex(model, opt_level="O2")

model.to("cuda")
criterion = nn.CrossEntropyLoss()

trainloader = DataLoader(
    MNIST(root="./", download=True, train=True, transform=ToTensor()),
    batch_size=32,
    num_workers=4,
)
for epoch in range(20):
    meter = ConfusionMatrix(10)
    trainloader_ = tqdm_(trainloader)
    for i, (img, target) in enumerate(trainloader_):
        img, target = img.cuda(), target.cuda()
        pred = net(img)
        loss = criterion(pred, target)
        with ZeroGradientBackwardStep(loss, model) as scaled_loss:
            scaled_loss.backward()

        meter.add(pred.max(1)[1], target)
        trainloader_.set_postfix(meter.summary())
