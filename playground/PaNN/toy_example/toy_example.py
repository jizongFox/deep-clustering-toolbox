# this is the toy example to optimize the primal-dual gradient descent.
# toy example for mnist dataset
import sys
from pathlib import Path

from torchvision import transforms

from deepclustering.model import Model

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

try:
    from .utils import get_prior_from_dataset, SimpleNet
except ImportError:
    from toy_example.utils import SimpleNet, get_prior_from_dataset
try:
    from .trainer import (
        SemiTrainer,
        SemiEntropyTrainer,
        SemiPrimalDualTrainer,
        SemiWeightedIICTrainer,
        SemiUDATrainer,
    )
except ImportError:
    from toy_example.trainer import (
        SemiTrainer,
        SemiEntropyTrainer,
        SemiPrimalDualTrainer,
        SemiWeightedIICTrainer,
        SemiUDATrainer,
    )
try:
    from .dataset import get_mnist_dataloaders
except ImportError:
    from toy_example.dataset import get_mnist_dataloaders
from deepclustering.optim import RAdam
from deepclustering.utils import fix_all_seed
from deepclustering.manager import ConfigManger
from torch.optim.lr_scheduler import MultiStepLR

config = ConfigManger("./config.yaml", integrality_check=False).config

fix_all_seed(0)
## dataloader part
unlabeled_class_sample_nums = {0: 10000, 1: 1000, 2: 2000, 3: 3000, 4: 4000}
dataloader_params = {
    "batch_size": 64,
    "num_workers": 1,
    "drop_last": True,
    "pin_memory": True,
}
train_transform = transforms.Compose([transforms.ToTensor()])
val_transform = transforms.Compose([transforms.ToTensor()])
labeled_loader, unlabeled_loader, val_loader = get_mnist_dataloaders(
    labeled_sample_num=10,
    unlabeled_class_sample_nums=unlabeled_class_sample_nums,
    train_transform=train_transform,
    val_transform=val_transform,
    dataloader_params=dataloader_params,
)
prior = get_prior_from_dataset(unlabeled_loader.dataset)
print("prior for unlabeled dataset", prior)
# network part
import warnings

fix_all_seed(int(config.get("Seed", 0)))

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    net = SimpleNet(1, len(unlabeled_class_sample_nums))
    optim = RAdam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = MultiStepLR(optim, milestones=[50, 80], gamma=0.2)
    model = Model()
    model.torchnet = net
    model.optimizer = optim
    model.scheduler = scheduler

# trainer part
Trainer = {
    "SemiTrainer": SemiTrainer,
    "SemiEntropyTrainer": SemiEntropyTrainer,
    "SemiPrimalDualTrainer": SemiPrimalDualTrainer,
    "SemiWeightedIICTrainer": SemiWeightedIICTrainer,
    "SemiUDATrainer": SemiUDATrainer,
}.get(config["Trainer"]["name"])
assert Trainer

trainer = Trainer(
    model,
    labeled_loader,
    unlabeled_loader,
    val_loader,
    prior=prior,
    config=config,
    **{k: v for k, v in config["Trainer"].items() if k != "name"}
)
trainer.start_training()
