"""
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from deepclustering.model import Model
from deepclustering.utils import DataIter
from playground.subspaceClustering.arch import ClusterNet5g
from torch import optim
from torch.utils.data import DataLoader

try:
    from .subclassClustering import SubSpaceClusteringMethod2, SubSpaceClusteringMethod
    from .dataset import TensorDataset, create_gaussian_norm_dataset
except:
    from subclassClustering import SubSpaceClusteringMethod2, SubSpaceClusteringMethod
    from dataset import TensorDataset, create_gaussian_norm_dataset
arch_param = {
    "name": "clusternet5g",
    "input_size": 32,
    "num_channel": 1,
    "num_sub_heads": 1,
}
optim_dict = {"name": "Adam", "lr": 0.0001, "weight_decay": 1e-4}
scheduler_dict = {
    "name": "MultiStepLR",
    "milestones": [100, 200, 300, 400, 500, 600, 700, 800, 900],
    "gamma": 0.75,
}
model = Model(
    arch_dict=arch_param, optim_dict=optim_dict, scheduler_dict=scheduler_dict
)
arch_param.pop("name")
optim_dict.pop("name")
# override the module with self-defined models.
device = torch.device("cpu")
model.torchnet = ClusterNet5g(**arch_param)
model.optimizer = optim.Adam(model.torchnet.parameters(), **optim_dict)
model.scheduler.optimizer = model.optimizer

dataset = create_gaussian_norm_dataset(
    num_cluster=10, num_exp=50, num_feature=10, var=0.1, shuffle=False
)

dataset = TensorDataset(
    torch.Tensor(dataset), torch.Tensor(np.zeros_like(dataset[:, 0])[..., None])
)
dataloader = DataIter(DataLoader(dataset, shuffle=True, batch_size=50))

# previous MNIST dataset
# training_set = MNIST(root=DATA_PATH, train=True, download=True, transform=Compose([Resize(32), ToTensor()]))
# labeled_set = Subset(dcp(training_set), indices=list(range(200)))
# unlabeled_set = Subset(dcp(training_set), indices=list(range(200, 500)))
# labeled_loader = DataIter(DataLoader(labeled_set, batch_size=50, shuffle=True))
# unlabeled_loader = DataIter(DataLoader(unlabeled_set, batch_size=50, shuffle=True))
# val_loader = DataLoader(MNIST(DATA_PATH, train=False), batch_size=10, shuffle=False)

subspace_method = SubSpaceClusteringMethod(
    model=model, num_samples=500, lamda=0.1, lr=0.0005, device=device
)
model.to(device)

plt.ion()
for i, ((imgs, _), index) in enumerate(dataloader):
    subspace_method.set_input(imgs=imgs, index=index)
    subspace_method.update()
    if i % 1 == 0:
        plt.clf()
        plt.imshow(subspace_method.adj_matrix.cpu().__abs__())
        plt.colorbar()
        plt.title(
            f"{i}, zero={(subspace_method.adj_matrix.cpu().__abs__().numpy() <= 1e-4).sum()}"
        )
        plt.show()
        plt.pause(0.001)
    if i > 1e6:
        break
