"""Train CIFAR10 with PyTorch."""
from __future__ import print_function

import argparse
import os
import pickle
from random import random

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from models import *
from torch.autograd import Variable
from utils import progress_bar

# real multistep
# Optimization as a model for few shot learning
# How to train your MAML

# 5 sec stoch
# 24 sec discr
#

parser = argparse.ArgumentParser(
    description="PyTorch Differentiable Discrete Augmented Training"
)
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument(
    "--dlr", default=0.1, type=float, help="data augmentation learning rate multiplier"
)
parser.add_argument("--name", "-n", type=str, default="", help="experiment name")
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
parser.add_argument("--mscale", "-m", type=str, default="sconv", help="use multiscales")
parser.add_argument("--dataset", "-d", default="CIFAR10", type=str, help="dataset name")
parser.add_argument("--batchsize", "-b", type=int, default=200, help="batch size")
parser.add_argument(
    "--gradit",
    "-g",
    type=int,
    default=1,
    help="Number of gradient itereations in the past",
)
parser.add_argument(
    "--whengrad", type=int, default=1, help="How often update the gradient"
)
parser.add_argument(
    "--factor", "-f", type=float, default=0.5, help="Initial value for factor"
)
parser.add_argument(
    "--kchannels", "-k", type=int, default=1, help="Multiplier of used channels "
)
parser.add_argument(
    "--schedule1",
    type=int,
    default=200,
    help="Epoch to reduce schedule by 10, -1 to not use",
)
parser.add_argument(
    "--smpuniform",
    type=int,
    default=-1,
    help="Sample uniform for estimation of weights every X times and the rest is based on --sampling",
)
parser.add_argument(
    "--schedule2",
    type=int,
    default=400,
    help="Second epoch to reduce schedule by 10, -1 to not use",
)
parser.add_argument(
    "--sampling",
    type=float,
    default=0.5,
    help="Sampling ratio: 0=uniform 1=greedy(wont work)",
)
parser.add_argument("--stoch", "-s", action="store_true", help="Stochastic")
parser.add_argument(
    "--sametr",
    action="store_true",
    help="Use the same transformation for the netire mini-batch. Around 50% faster but poorer estimation of weights",
)
parser.add_argument("--useadam", "-a", action="store_true", help="Use Adam optimizer")
parser.add_argument("--dense", "--dense", action="store_true", help="Dense seq")
parser.add_argument(
    "--learnCfactors", "-l", action="store_true", help="Learn Continuous Factors"
)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.dataset == "CIFAR10":
    isize = 32
elif args.dataset == "STL10":
    isize = 48
elif args.dataset == "MNIST":
    isize = 28

torch.manual_seed(3)

from torchvision import datasets, transforms
from torch.utils.data import Dataset


class CIFAR10ID(Dataset):
    def __init__(self, root, train, download, transform):
        self.cifar10 = datasets.CIFAR10(
            root="./data", download=download, train=train, transform=transform
        )

    def __getitem__(self, index):
        data, target = self.cifar10[index]

        # Your transformations here (or set it in CIFAR10)

        return data, target, index

    def __len__(self):
        return len(self.cifar10)


# dataset = MyDataset()
# loader = DataLoader(dataset,
#                    batch_size=1,
#                    shuffle=True,
#                    num_workers=1)

# for batch_idx, (data, target, idx) in enumerate(loader):
#    print('Batch idx {}, dataset index {}'.format(
#        batch_idx, idx))


# Currently there is a risk of dropping all paths...
# We should create a version that take all paths into account to make sure one stays alive
# But then keep_prob is meaningless and we have to copute/keep track of the conditional probability
class DropPath(nn.Module):
    def __init__(self, module, keep_prob=0.9):
        super(DropPath, self).__init__()
        self.module = module
        self.keep_prob = keep_prob
        self.shape = None
        self.training = True
        self.dtype = torch.FloatTensor

    def forward(self, *input):
        if self.training:
            # If we don't now the shape we run the forward path once and store the output shape
            if self.shape is None:
                temp = self.module(*input)
                self.shape = temp.size()
                if temp.data.is_cuda:
                    self.dtype = torch.cuda.FloatTensor
                del temp
            p = random()
            if p <= self.keep_prob:
                return Variable(self.dtype(self.shape).zero_())
            else:
                return self.module(*input) / self.keep_prob  # Inverted scaling
        else:
            return self.module(*input)


class Affine(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, angle, translate, scale, shear, resample=0, fillcolor=None):
        # assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.angle = angle
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.resample = resample
        self.fillcolor = fillcolor

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        return torchvision.transforms.functional.affine(
            img,
            self.angle,
            self.translate,
            self.scale,
            self.shear,
            self.resample,
            self.fillcolor,
        )


#    def __repr__(self):
#        interpolate_str = _pil_interpolation_to_str[self.interpolation]
#        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


# Data

print("==> Preparing data..")
transform_train = transforms.Compose(
    [
        # transforms.RandomCrop(isize, padding=isize/4),#8),
        transforms.RandomHorizontalFlip(),
        # Affine(angle=np.pi, translate=(10,20), scale=0.8, shear=0, resample=False, fillcolor=0),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_valtest = transforms.Compose(
    [
        # transforms.RandomCrop(isize, padding=isize/4),#8),
        transforms.RandomHorizontalFlip(),
        # Affine(angle=90, translate=(0,0), scale=1, shear=0, resample=False, fillcolor=0),
        # transforms.RandomAffine(degrees=(0,90), translate=(0,0), scale=(1,2), shear=None, resample=False, fillcolor=0),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        # transforms.RandomCrop(isize, padding=isize/4),#8),
        # transforms.functional.affine(img, angle, translate, scale, shear, resample=0, fillcolor=None)
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

if args.dataset == "MNIST":
    transform_train = transforms.Compose(
        [
            # transforms.RandomCrop(isize, padding=isize/4),#8),
            # transforms.RandomHorizontalFlip(),
            # Affine(angle=np.pi, translate=(10,20), scale=0.8, shear=0, resample=False, fillcolor=0),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    transform_valtest = transforms.Compose(
        [
            # transforms.RandomCrop(isize, padding=isize/4),#8),
            # Affine(angle=90, translate=(0,0), scale=1, shear=0, resample=False, fillcolor=0),
            transforms.RandomAffine(
                degrees=(0, 0),
                translate=(0, 0),
                scale=(1, 1),
                shear=None,
                resample=False,
                fillcolor=0,
            ),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    transform_test = transforms.Compose(
        [
            # transforms.RandomCrop(isize, padding=isize/4),#8),
            # transforms.functional.affine(img, angle, translate, scale, shear, resample=0, fillcolor=None)
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    trainset1 = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=transform_train
    )
    valtestset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transform_valtest
    )
    testset = torchvision.datasets.MNIST(
        "./data", train=False, transform=transform_test
    )
elif args.dataset == "CIFAR10":
    trainset1 = CIFAR10ID(
        root="./data", train=True, download=True, transform=transform_train
    )
    valtestset = CIFAR10ID(
        root="./data", train=False, download=True, transform=transform_valtest
    )
    testset = CIFAR10ID(
        root="./data", train=False, download=True, transform=transform_test
    )
elif args.dataset == "STL10":
    trainset1 = torchvision.datasets.STL10(
        root="./data", split="test", download=True, transform=transform_train
    )
    valtestset = torchvision.datasets.STL10(
        root="./data", split="train", download=True, transform=transform_valtest
    )
    testset = torchvision.datasets.STL10(
        root="./data", split="train", download=True, transform=transform_test
    )
train_size = int(0.8 * len(trainset1))
val_size = len(trainset1) - train_size
# total_size = 0.1
trainset, valset = torch.utils.data.random_split(trainset1, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batchsize, shuffle=True, num_workers=2, drop_last=True
)
trainloader2 = torch.utils.data.DataLoader(
    trainset, batch_size=args.batchsize, shuffle=True, num_workers=2
)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=args.batchsize, shuffle=True, num_workers=2, drop_last=True
)
valtestloader = torch.utils.data.DataLoader(
    valtestset, batch_size=args.batchsize, shuffle=True, num_workers=2
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batchsize, shuffle=False, num_workers=2
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Model
print("==> Building model..")
# net = VGG('VGG19')
# net = LeNet()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()


from torch.distributions import *


class signout(nn.Module):
    def forward(self, x):
        mask = 0.1 * torch.ones(x.shape, dtype=x.dtype, device=x.device)
        mask = (Bernoulli(mask).sample() - 0.5) * 2
        return x * mask


class Transf(nn.Module):
    def __init__(
        self,
        position,
        factor=1,
        base=[1, 0, 0, 0, 1, 0],
        distr="uniform",
        learn_factor="No",
    ):
        super(Transf, self).__init__()
        self.affine = torch.tensor(base, device="cuda").float()
        self.position = position
        self.learn_factor = learn_factor
        if learn_factor == "No":
            self.tfactor = [factor]
            if position == -1:
                self.tfactor = (torch.ones(6) * factor).cuda()
        elif learn_factor == "Yes":
            if position == -1:
                self.tfactor = nn.Parameter(torch.ones(6) * factor)
            else:
                self.tfactor = nn.Parameter(torch.ones(1) * factor)
        elif learn_factor == "Yes+Bias":
            if position == -1:
                self.tfactor = nn.Parameter(torch.ones(12) * factor)
                self.tfactor[6:] = 0
            else:
                self.tfactor = nn.Parameter(torch.ones(2) * factor)
                self.tfactor[1] = 0
        self.distr = distr  # uniform or normal

    def forward(self, x):
        # x is a random vector
        dx = torch.zeros(x.shape[0], 6).cuda()
        if self.position == -1:
            dx[:, :] = x.view(-1, 1) * self.tfactor[:6].view(1, 6)
            if self.learn_factor == "Yes+Bias":
                dx[:, :] = dx[:, :] + selfs.tfactor[6:]
        else:
            dx[:, self.position] = x.view(-1) * self.tfactor[0]
            if self.learn_factor == "Yes+Bias":
                dx[:, self.position] = dx[:, self.position] + self.tfactor[1]

        return (self.affine.view(1, 6) + dx).view(-1, 6)


class Dataug(nn.Module):
    def __init__(self, transf=[], sampling=0.5):
        super(Dataug, self).__init__()
        if transf == []:
            self.transf = [
                [1, 0, 0, 0, 1, 0],  # identity
                [0.9, 0, 0, 0, 1, 0],  # scale 0.9 x
                [1, 0, 0, 0, 0.9, 0],  # scale 0.9 y
                [1.1, 0, 0, 0, 1, 0],  # scale 1.1 x
                [1, 0, 0, 0, 1.1, 0],  # scale 1.1 y
            ]
        else:
            self.transf = nn.ModuleList(transf)
        self.fc = sampling
        self.factor = nn.Parameter(
            1 * torch.ones(len(transf))
        )  # ,torch.ones(len(transf)))
        self.counter = 0

    def forward(self, x):
        bs = x.shape[0]
        ntr = len(self.transf)
        uniform = Uniform(low=-torch.ones(bs, 1).cuda(), high=torch.ones(bs, 1).cuda())
        normal = Normal(loc=torch.zeros(bs, 1).cuda(), scale=torch.ones(bs, 1).cuda())
        usmp = uniform.sample()
        nsmp = normal.sample()
        self.counter += 1
        if args.stoch:
            theta = torch.zeros(bs, 6).cuda()
            newx = torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
            # self.fc = 0.5
            if args.smpuniform > 0 and (self.counter % args.smpuniform) == 0:
                self.qsmp = torch.ones(
                    1, len(self.factor), device=torch.device("cuda")
                ).softmax(1)
            else:
                self.qsmp = (
                    (self.fc) * self.factor.view(1, -1)
                    + (1 - self.fc) / len(self.factor)
                ).softmax(1)
            if not (args.sametr):
                cat = Categorical(
                    probs=torch.ones((bs, len(self.factor))).cuda() * self.qsmp
                )  # *self.factor.view(1,-1))
                self.tsmp = cat.sample()
                for idtr, tr in enumerate(self.transf):
                    sel = self.tsmp == idtr
                    if tr.distr == "uniform":
                        theta[sel, :] = tr(usmp[sel])
                    elif tr.distr == "normal":
                        theta[sel, :] = tr(nsmp[sel])
            else:  # 50% faster, but worse performance...
                cat = Categorical(probs=self.qsmp)
                self.tsmp = torch.mm(
                    torch.ones(bs, 1).cuda(), cat.sample().float().view(1, 1)
                ).long()
                sel = self.tsmp[0]
                tr = self.transf[sel]
                if tr.distr == "uniform":
                    theta = tr(usmp)
                elif tr.distr == "normal":
                    theta = tr(nsmp)

            # import numpy as np
            # print(np.histogram(self.tsmp.cpu().numpy(),bins=3)[0])
            grid = F.affine_grid(theta.view(-1, 2, 3), x.size())
            newx = F.grid_sample(x, grid)
            self.theta = theta.view(-1, 2, 3)
        else:
            theta = torch.zeros(ntr, bs, 6).cuda()
            newx = torch.zeros(ntr, x.shape[0], x.shape[1], x.shape[2], x.shape[3])
            for idtr, tr in enumerate(self.transf):
                if tr.distr == "uniform":
                    theta[idtr, :, :] = tr(usmp)
                elif tr.distr == "normal":
                    theta[idtr, :, :] = tr(nsmp)
                grid = F.affine_grid(theta[idtr].view(-1, 2, 3), x.size())
                newx[idtr] = F.grid_sample(x, grid)
            self.theta = theta.view(-1, 2, 3)
            newx = newx.view(-1, x.shape[1], x.shape[2], x.shape[3])

        if len(self.buffer) > 0:
            if self.buffer[0].shape[0] == newx.shape[0]:
                self.buffer.append(theta.detach().cpu().numpy())
        else:
            self.buffer.append(theta.detach().cpu().numpy())
        return newx


class Wloss(nn.Module):
    def __init__(self, factor):  # ,transf=[1,0,0,0,1,0]):
        super(Wloss, self).__init__()
        self.factor = factor  # nn.Parameter(1*torch.ones(n))

    def forward(self, outputs, targets, tsmp=[], q=[]):
        if args.stoch:
            smax = self.factor.softmax(0)
            negreward = -outputs.log_softmax(1)[
                torch.arange(outputs.shape[0]).to(torch.long), targets
            ]
            loss = (negreward * smax[tsmp] / (len(smax) * (q[0, tsmp]))).mean()
            # loss = (negreward*smax[tsmp]).mean()
            # print(loss,(negreward*smax[tsmp]).mean())
            # dfdg
        # to visualize computational graph
        # from torchviz import make_dot
        # graph=make_dot(loss)
        # graph.view()
        # rward = (outputs.argmax(1)==targets).float()
        # loss = (negreward-((rward-rward.mean())*(smax[tsmp]).log())).mean()
        # loss = -((outputs.argmax(1)==targets).float()*prob.log()).mean()
        # print(negreward.mean(),-((outputs.argmax(1)==targets).float()*prob.log()).mean())
        # print(loss)
        # sfs
        else:
            smax = self.factor.softmax(0)
            norm = smax
            fweight = (
                torch.mm(
                    norm.view(-1, 1),
                    torch.ones(len(outputs) / len(self.factor)).cuda().view(1, -1),
                )
                .view(-1)
                .cuda()
            )
            loss = (
                -outputs.log_softmax(1)[
                    torch.arange(outputs.shape[0]).to(torch.long), targets
                ]
                * fweight
            ).mean()

        return loss


class Prediction(nn.Module):
    def __init__(self, samples, classes):  # ,transf=[1,0,0,0,1,0]):
        super(Prediction, self).__init__()
        self.factorpred = nn.Parameter(
            torch.ones(len(trainset) + len(valset), classes).cuda()
        )

    def forward(self, outputs, smpid):
        loss = (
            (
                -outputs.log_softmax(1)[torch.arange(outputs.shape[0])]
                * self.factorpred[smpid].softmax(1)
            )
            .sum(1)
            .mean()
        )
        return loss


net = LeNet(k=args.kchannels)
# from models import resnet
# net = resnet.ResNet18()
# net = ResNet18()

wcriterion = Prediction(len(trainset), 10)  #
criterion = nn.CrossEntropyLoss()
allnames = [x for x, y in net.named_parameters()] + [
    x for x, y in wcriterion.named_parameters()
]
allparam = [y for x, y in net.named_parameters()] + [
    y for x, y in wcriterion.named_parameters()
]

netidx = [idx for idx, x in enumerate(allnames) if x.find("factor") is -1]
# alphaidx = [idx for idx,x in enumerate(allnames) if x.find('factor') is not -1]
# alphaidx = [idx for idx,x in enumerate(wcriterion.parameters())]

netparam = [allparam[idx] for idx in netidx]
alphaparam = [x for x in wcriterion.parameters()]  # [allparam[idx] for idx in alphaidx]

# optimizernet = optim.SGD(netparam, lr=args.lr, momentum=0.9, weight_decay=0)#-5e-4)
if len(alphaparam) > 0:
    if args.useadam:
        optimizeralphaval = optim.Adam(
            alphaparam, lr=args.lr * args.dlr
        )  # , momentum=0.0, weight_decay=0.0)#-5e-4)
    else:
        optimizeralphaval = optim.SGD(
            alphaparam, lr=args.lr * args.dlr, momentum=0.0, weight_decay=0.0
        )  # -5e-4)

net = net.to(device)
wcriterion = wcriterion.to(device)
if device == "cuda":
    net = torch.nn.DataParallel(net)
    wcriterion = torch.nn.DataParallel(wcriterion)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load("./checkpoint/%s.chk" % args.name)
    net.load_state_dict(checkpoint["net"])
    wcriterion.load_state_dict(checkpoint["wcri"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    f = open("results/%s" % args.name, "r")
    log = pickle.load(f)
    f.close()

import matplotlib.pyplot as plt
import numpy as np


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# show images
# imshow(torchvision.utils.make_grid(images))

usetestforval = False
usetrainforval = False
real_lr = args.lr


# Training
def train(epoch, netparam, usetestforval, moms, log):
    print("\nEpoch: %d" % epoch)
    net.train()
    lmb = 0.0
    train_loss = 0
    correct = 0
    total = 0
    rel = 1
    nit_grad = args.gradit
    when_grad = args.whengrad
    if usetestforval:
        valitem = iter(valtestloader)  # (valtestloader)
    else:
        if usetrainforval:
            valitem = iter(trainloader2)  # (valtestloader)
        else:
            valitem = iter(valloader)  # (valtestloader)
    optimizeralphaval.zero_grad()
    # aug.module.buffer=[]
    fast_weights = [{"old": [], "new": [x for x in netparam]}]
    # fast_weights.append([x for x in netparam])
    pbuf = 0

    for batch_idx, (inputs, targets, smpid) in enumerate(trainloader):
        # if batch_idx==0:

        inputs, targets = inputs.to(device), targets.to(device)
        # inputs = (inputs-inputs.mean())/inputs.std()

        weights = [nn.Parameter(x.detach()) for x in fast_weights[pbuf]["new"]]
        # for idx,x in enumerate(weights):
        #    x.require_grad = True
        #    x.retain_grad()
        # [x.require_grad = True for x in weights]
        outputs = net(inputs, weights)

        nit = 1
        if args.stoch:
            loss = wcriterion(outputs, smpid)
        else:
            targets = targets.repeat(len(aug.module.transf))
            loss = wcriterion(outputs, targets)

        grads = torch.autograd.grad(loss, weights, create_graph=True, retain_graph=True)
        moms = [0.9 * m + grad for (m, grad) in zip(moms, grads)]

        fast_weights.append({"old": weights, "new": []})
        fast_weights[pbuf + 1]["new"] = [
            x - real_lr * m for (x, m) in zip(weights, moms)
        ]

        # fast_weights[pbuf+1]['new'] = [x - args.lr*g for (x,g) in zip(weights,grads)]

        pbuf += 1

        if pbuf > nit_grad:
            pbuf = nit_grad
            del fast_weights[0]

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx,
            len(trainloader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )

        if len(alphaparam) > 0:
            if batch_idx % when_grad == 0:
                try:
                    (inputs2, targets2, smpid) = valitem.next()
                except (StopIteration):
                    if usetestforval:
                        valitem = iter(valtestloader)  # (valtestloader)
                    else:
                        if usetrainforval:
                            valitem = iter(trainloader2)  # (valtestloader)
                        else:
                            valitem = iter(valloader)  # (valtestloader)
                    (inputs2, targets2, smpid) = valitem.next()

                # inputs2, targets2 = inputs, targets
                inputs2, targets2 = inputs2.to(device), targets2.to(device)
                # inputs2 = (inputs2-inputs2.mean())/inputs2.std()
                # if batch_idx%200==0:
                #    plt.figure(3)
                #    imshow(torchvision.utils.make_grid(inputs2[:64].cpu()))
                #    plt.pause(0.01)
                # optimizeralphaval.zero_grad()
                outputs = net(inputs2, fast_weights[pbuf]["new"])

                # def hook(module, in_grad, out_grad):
                #   sfsdf

                # fast_weights[pbuf][0].register_backward_hook(hook)

                loss2 = criterion(outputs, targets2)  # ,aug.module.tsmp)
                loss2.backward(retain_graph=True)
                for l in range(len(fast_weights) - 1):
                    curr_grad = [x.grad for x in fast_weights[-l - 1]["old"]]
                    # curr_grad = torch.autograd.grad(loss2, fast_weights[pbuf]['old'], create_graph=True, retain_graph = True)
                    [
                        p.backward(curr_grad[idp], retain_graph=True)
                        for idp, p in enumerate(fast_weights[-l - 2]["new"])
                    ]

                optimizeralphaval.step()
                optimizeralphaval.zero_grad()
                for x in moms:
                    x.detach_()

                for idl, param in enumerate(netparam):
                    param.data = fast_weights[pbuf]["new"][idl]
                # if param.grad!=None:
                # param.grad[:] = 0
                # param=param.detach()

    optimizeralphaval.step()

    # for idp,params in enumerate(netparam):
    #    params[:] = fast_weights[idp]

    # tot = float(net.module.f1,net.module.f2,net.module.f3)
    #    print(net.module.f)
    #    print(net.module.length)
    #    print(net.module.distc)
    #    net.module.reset()
    # net.module.set_factor(1.1)
    # print(aug.module.factor.data.cpu().numpy())
    log["train"].append(train_loss / (batch_idx + 1))
    log["train_acc"].append(100.0 * correct / total)
    # log['weights'].append(wcriterion.factorpred.softmax(0).cpu().detach().numpy())
    # log['params'].append([x.cpu().detach().numpy() for x in aug.module.parameters()])


#    if 1:
# print(np.array(aug.module.buffer).reshape((-1,2,3)).mean(0))
# print(np.array(aug.module.buffer).reshape((-1,2,3)).std(0))
# print(wcriterion.module.factorpred.softmax(0))
# print([x for x in aug.module.parameters()])
# print(aug.module.factor1.weight)

# Training
def val(epoch, usetestforval, log):
    print("\nEpoch: %d" % epoch)
    net.train()
    lmb = 0.2
    train_loss = 0
    correct = 0
    total = 0
    if usetestforval:
        loader = valtestloader
    else:
        loader = valloader
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        # inputs = (inputs-inputs.mean())/inputs.std()
        optimizeralphaval.zero_grad()
        outputs = net(aug(inputs))
        if args.stoch:
            if 1:
                loss = (
                    -lmb
                    * (
                        (torch.stack(net.module.smp).mean()).log()
                        * (1 - (outputs.max(1)[1].eq(targets)).float())
                    ).mean()
                )
                # loss = -outputs.softmax(1)[torch.arange(outputs.shape[0]).to(torch.long),targets].log().mean() -lmb*((torch.stack(net.module.smp).mean()).log()*(1-(outputs.max(1)[1].eq(targets)).float())).mean()
                # loss = (-(net.module.smp1+net.module.smp2+net.module.smp3).log()*outputs.softmax(1)[torch.arange(outputs.shape[0]).to(torch.long),targets].log()).mean()
            else:
                loss = (
                    -(net.module.smp1 + net.module.smp2 + net.module.smp3).mean(1)
                    * outputs.softmax(1)[
                        torch.arange(outputs.shape[0]).to(torch.long), targets
                    ].log()
                ).mean()
        else:
            loss = (
                -outputs.softmax(1)[
                    torch.arange(outputs.shape[0]).to(torch.long), targets
                ]
                .log()
                .mean()
            )
            # loss = criterion(outputs,targets)
        loss.backward()
        optimizeralphaval.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx,
            len(valloader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )
    # tot = float(net.module.f1,net.module.f2,net.module.f3)
    #    print(net.module.f)
    #    print(net.module.length)
    #    print(net.module.distc)
    #    net.module.reset()
    # net.module.set_factor(1.1)
    log["val"].append(train_loss / (batch_idx + 1))
    log["val_acc"].append(100.0 * correct / total)
    print(aug.module.factor.item())
    for l in net.modules():
        # print(type(l))
        # raw_input()
        if type(l) == sConv2d:
            print(l.filter)
            print(l.nalpha.cpu().detach().numpy())
            # print(l.alpha.cpu().detach().numpy())
    # raw_input()


def test_val(epoch, usetestforval, log):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    if usetestforval:
        loader = valtestloader
    else:
        loader = valloader
    with torch.no_grad():
        for batch_idx, (inputs, targets, idx) in enumerate(loader):  # (valtestloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # inputs = (inputs-inputs.mean())/inputs.std()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(testloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )
    log["val"].append(test_loss / (batch_idx + 1))
    log["val_acc"].append(100.0 * correct / total)
    print("\n")


def test(epoch, log):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, idx) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # inputs = (inputs-inputs.mean())/inputs.std()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(testloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )

    log["test"].append(test_loss / (batch_idx + 1))
    log["test_acc"].append(100.0 * correct / total)
    # Save checkpoint.
    acc = 100.0 * correct / total
    if acc > best_acc:
        print("Saving..")
        state = {
            "net": net.state_dict(),
            "wcri": wcriterion.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        torch.save(state, "./checkpoint/%s.chk" % args.name)
        best_acc = acc


moms = [torch.zeros(x.shape).cuda() for x in netparam]

try:
    log
except:
    log = {
        "train_acc": [],
        "val_acc": [],
        "test_acc": [],
        "train": [],
        "val": [],
        "test": [],
        "weights": [],
        "params": [],
        "args": args,
    }

for epoch in range(start_epoch, start_epoch + 1000):
    train(epoch, netparam, usetestforval, moms, log)
    test_val(epoch, usetestforval, log)
    test(epoch, log)
    if epoch == args.schedule1 or epoch == args.schedule2:
        real_lr *= 0.1
        for g in optimizeralphaval.param_groups:
            g["lr"] *= 0.1
    # scheduler.step()
    # scheduleralpha.step()
    # save
    f = open("results/%s" % args.name, "w")
    pickle.dump(log, f)
    f.close()
