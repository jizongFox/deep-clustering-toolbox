from swa_trainer import SWATrainer
from deepclustering import DATA_PATH
from deepclustering.dataset.classification.cifar import CIFAR10
from torch.utils.data import DataLoader
from deepclustering.augment.pil_augment import ToTensor
from deepclustering.manager import ConfigManger
from deepclustering.model import Model
from pathlib import Path

DEFAULT_CONFIG = str(Path(__file__).parent / "config.yaml")

config = ConfigManger(DEFAULT_CONFIG_PATH=DEFAULT_CONFIG, verbose=True).config
# create model:
model = Model(
    arch_dict=config["Arch"],
    optim_dict=config["Optim"],
    scheduler_dict=config["Scheduler"],
)
train_loader = DataLoader(CIFAR10(root=DATA_PATH, transform=ToTensor(), train=True), **config["DataLoader"])
val_loader = DataLoader(CIFAR10(root=DATA_PATH, transform=ToTensor(), train=False), **config["DataLoader"])

trainer = SWATrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    **config["Trainer"]
)
trainer.start_training()
