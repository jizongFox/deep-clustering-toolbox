"""
A simple example to show if the apex works
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim import Adam
from deepclustering.model import Model, ZeroGradientBackwardStep, to_Apex
from deepclustering.meters import ConfusionMatrix
from deepclustering.utils import nice_dict, tqdm_

plt.ion()
from deepclustering.dataset.segmentation.toydataset import Cls_ShapesDataset

train_set = Cls_ShapesDataset(max_object_scale=0.75)
train_loader = DataLoader(train_set, batch_size=4, num_workers=4, shuffle=True)

val_set = Cls_ShapesDataset(max_object_scale=0.4)
val_loader = DataLoader(val_set, batch_size=4, num_workers=4)


def conv_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_block1 = conv_block(3, 16, 7, 2, 3)
        self.conv_block2 = conv_block(16, 32, 3, 1, 1)
        self.conv_block3 = conv_block(32, 64, 3, 1, 1)
        self.conv_block4 = conv_block(64, 128, 3, 1, 1)
        self.conv_block5 = conv_block(128, 128, 3, 1, 1)
        self.fc = nn.Linear(128, 3)
        self.down_sample = nn.MaxPool2d((2, 2), stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        out = self.conv_block1(input)
        out = self.down_sample(self.conv_block2(out))
        out = self.down_sample(self.conv_block3(out))
        out = self.down_sample(self.conv_block4(out))
        out = self.down_sample(self.conv_block5(out))
        out = self.avg_pool(out).view(out.size(0), -1)
        out = self.fc(out)
        return out


net = Net()
net.cuda()
optimiser = Adam(net.parameters(), lr=1e-4)
model = Model()
model.torchnet = net
model.optimizer = optimiser
model = to_Apex(model, opt_level='O2')

criterion = nn.CrossEntropyLoss()
model.to(torch.device('cuda'))


def val(model, val_loader, epoch):
    model.eval()
    acc_meter = ConfusionMatrix(num_classes=3)
    val_loader_ = tqdm_(val_loader)
    val_loader_.set_description(f'Validating {epoch}')
    for i, (img, target) in enumerate(val_loader_):
        img, target = img.cuda(), target.cuda()
        pred = model(img)
        acc_meter.add(pred.max(1)[1], target)
        val_loader_.set_postfix(acc_meter.summary())
    model.train()
    print(f"Validating epoch: {epoch}: {nice_dict(acc_meter.summary())}")


for epoch in range(1000):
    train_loader_ = tqdm_(train_loader, leave=False)
    acc_meter = ConfusionMatrix(num_classes=3)

    for i, (img, target) in enumerate(train_loader_):
        img, target = img.cuda(), target.cuda()
        pred = model(img)
        loss = criterion(pred, target)
        with ZeroGradientBackwardStep(loss, model) as scaled_loss:
            scaled_loss.backward()
        acc_meter.add(pred.max(1)[1], target)
        train_loader_.set_postfix(acc_meter.summary())
    print(f"  Training epoch: {epoch}: {nice_dict(acc_meter.summary())}")
    with torch.no_grad():
        val(model=model, val_loader=val_loader, epoch=epoch)
