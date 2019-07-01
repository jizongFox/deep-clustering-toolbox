#
#
#   file description
#
#
__author__ = "Jizong Peng"
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from deepclustering.utils import tqdm, tqdm_
from deepclustering.utils.classification.assignment_mapping import (
    hungarian_match,
    flat_acc,
    original_match,
)
from deepclustering.loss.IID_losses import IIDLoss, IID_loss
from deepclustering.loss.loss import KL_div, Entropy
from deepclustering.loss.IMSAT_loss import MultualInformaton_IMSAT
from sklearn.datasets import make_classification


class Net(nn.Module):
    def __init__(self, input_channel=5):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_channel, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=50)
        self.fc3 = nn.Linear(50, 10)

        self.fc4 = nn.Linear(50, 50)

    def forward(self, input):
        out = self.fc1(input)
        out = F.leaky_relu(out)
        out = self.fc2(out)
        out = F.leaky_relu(out)
        out1 = F.softmax(self.fc3(out), dim=1)
        # out2 = F.softmax(self.fc4(out), dim=1)

        return out1


net = Net(10).cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)


class Trainer:
    def __init__(self, model, optimizer) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer

    def training(self):
        x1, y = make_classification(1000, n_features=10, n_informative=5, n_classes=10)
        x1 = torch.from_numpy(x1).cuda().float()
        y = torch.from_numpy(y).cuda().long()
        itera: tqdm = tqdm_(range(100000))
        for i in itera:
            noise = torch.randn_like(x1).cuda()
            x2 = x1 + 0.1 * noise
            p1 = self.model(x1)
            p2 = self.model(x2)
            loss = self._loss_function(x1, p1, x2, p2)
            reordered, _ = hungarian_match(p1.max(1)[1], y, 10, 10)
            print(reordered.unique())
            acc = flat_acc(y, reordered)
            acc2 = flat_acc(y, p1.max(1)[1])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if i % 10 == 0:
                self.show(p1, p2)
            itera.set_postfix({"loss": loss.item(), "acc": acc, "acc2": acc2})

    def show(self, p1, p2):
        joint_p = p1.unsqueeze(1) * p2.unsqueeze(2)
        plt.figure(1)
        plt.clf()
        plt.imshow(joint_p.mean(0).detach().cpu())
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.001)

    def _loss_function(self, x1, p1, x2, p2):
        return 0


class IIC_Trainer(Trainer):
    def _loss_function(self, x1, p1, x2, p2):
        loss, *_ = IIDLoss()(p1, p2)
        marginal_entropy = Entropy()(p1.mean(0).unsqueeze(0)).mean()
        return loss - 0.5 * marginal_entropy


class IMSAT(Trainer):
    def _loss_function(self, x1, p1, x2, p2):
        loss, *_ = MultualInformaton_IMSAT()(p1)
        sat_loss = KL_div(reduce=True)(p2, p1.detach())
        return -loss + 0.1 * sat_loss


trainer = IIC_Trainer(net, optimizer)
trainer.training()
