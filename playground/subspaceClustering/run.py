"""
"""
from copy import deepcopy as dcp

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor, Compose, Resize

from deepclustering import DATA_PATH
from deepclustering.dataset.classification.mnist import MNIST
from deepclustering.model import Model
from deepclustering.utils import DataIter
from playground.subspaceClustering.arch import ClusterNet5g

try:
    from .subclassClustering import SubSpaceClusteringMethod2
except:
    from subclassClustering import SubSpaceClusteringMethod2
arch_param = {
    'name': 'clusternet5g',
    'input_size': 32,
    'num_channel': 1,
    'num_sub_heads': 1
}
optim_dict = {
    'name': 'Adam',
    'lr': 0.0001,
    'weight_decay': 1e-4
}
scheduler_dict = {
    'name': 'MultiStepLR',
    'milestones': [100, 200, 300, 400, 500, 600, 700, 800, 900],
    'gamma': 0.75
}
model = Model(
    arch_dict=arch_param,
    optim_dict=optim_dict,
    scheduler_dict=scheduler_dict
)
arch_param.pop('name')
optim_dict.pop('name')
# override the module with self-defined models.
model.torchnet = ClusterNet5g(**arch_param)
model.optimizer = optim.Adam(model.torchnet.parameters(), **optim_dict)
model.scheduler.optimizer = model.optimizer
training_set = MNIST(root=DATA_PATH, train=True, transform=Compose([Resize(32), ToTensor()]))
labeled_set = Subset(dcp(training_set), indices=list(range(20)))
unlabeled_set = Subset(dcp(training_set), indices=list(range(20, 50)))
labeled_loader = DataIter(DataLoader(labeled_set, batch_size=10, shuffle=True))
unlabeled_loader = DataIter(DataLoader(unlabeled_set, batch_size=10, shuffle=True))
val_loader = DataLoader(MNIST(DATA_PATH, train=False), batch_size=10, shuffle=False)
subspace_method = SubSpaceClusteringMethod2(model=model, num_samples=100,lamda=0.05)
model.to(torch.device('cuda'))


def merge(*args, dim=0):
    return torch.cat(args, dim=dim)


def create_gaussian_norm_dataset(num_cluster: int, num_exp: int, num_feature: int, var: float,
                                 shuffle=True) -> np.ndarray:
    """
    :param num_cluster:  Number of clusters
    :param num_exp: Number of examples per cluster
    :param num_feature: Number of features per example
    :param var: Variance of each cluster
    :return: numpy dataset
    """
    dataset = np.zeros((num_cluster * num_exp, num_feature))
    centers: np.ndarray = np.random.randn(num_cluster, num_feature)  # :shape  num_cluster * num_feature
    batch_num = 0
    for c in centers:
        for _ in range(num_exp):
            dataset[batch_num] = c + var * np.random.randn(*c.shape)
            batch_num += 1
    assert batch_num == num_cluster * num_exp
    if shuffle:
        np.random.shuffle(dataset)  # inplace operation
    return dataset


dataset = create_gaussian_norm_dataset(2, 50, 2, 0.1, shuffle=False)

plt.ion()
for i, ((limg, lgt, lindex), (uimg, _, uindex)) in enumerate(zip(labeled_loader, unlabeled_loader)):
    # index, imgs = merge(lindex, uindex), merge(limg, uimg)
    imgs = dataset
    imgs = torch.Tensor(imgs).float().to('cuda')
    index = torch.Tensor(list(range(0,100))).long()
    subspace_method.set_input(imgs=imgs, index=index)
    subspace_method.update()
    plt.clf()
    plt.imshow(subspace_method.adj_matrix.cpu().__abs__())
    plt.colorbar()
    plt.title(f"{i}, zero={(subspace_method.adj_matrix.cpu().__abs__().numpy() <=1e-4).sum()}")
    plt.show()
    plt.pause(0.001)

    if i > 1e4:
        break
