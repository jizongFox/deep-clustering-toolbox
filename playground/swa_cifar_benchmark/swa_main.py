from pathlib import Path

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms

from deepclustering import DATA_PATH
from deepclustering.dataloader.dataset import random_split
from deepclustering.dataset.classification.cifar import CIFAR10
from deepclustering.manager import ConfigManger
from deepclustering.model import Model
from playground.swa_cifar_benchmark.arch import _register_arch
from playground.swa_cifar_benchmark.my_scheduler import CosineAnnealingLR_
from playground.swa_cifar_benchmark.trainer import SWATrainer, SGDTrainer

lr_scheduler.CosineAnnealingLR = CosineAnnealingLR_

_ = _register_arch
DEFAULT_CONFIG = str(Path(__file__).parent / "config.yaml")

config = ConfigManger(DEFAULT_CONFIG_PATH=DEFAULT_CONFIG, verbose=True).config
# create model:
model = Model(
    arch_dict=config["Arch"],
    optim_dict=config["Optim"],
    scheduler_dict=config["Scheduler"],
)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# here you may want to have different labeled/unlabeled/val sets for semi supervised learning.
train_set = CIFAR10(root=DATA_PATH, transform=transform_train, train=True)
labeled_set, _, _ = random_split(train_set, [4000, 41000, 5000])
train_loader = DataLoader(labeled_set, **config["DataLoader"])
val_loader = DataLoader(CIFAR10(root=DATA_PATH, transform=transform_test, train=False), **config["DataLoader"])

Trainer = {"sgd": SGDTrainer, "swa": SWATrainer}.get(config["Trainer"].get("name").lower())
assert Trainer

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    **{k: v for k, v in config["Trainer"].items() if k != "name"}
)
trainer.start_training()
trainer.clean_up(10)
