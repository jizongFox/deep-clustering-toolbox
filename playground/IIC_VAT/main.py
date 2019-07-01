###############################
#   This file is to create experiments using Cifar10 dataset of IIC setting (double head) with acc = ~60.0%
#   to verify whether the VAT can help better with IIC.
#   This experiment can be long and a pretrained checkpoint can be used to reduce the time.
##############################
from pathlib import Path

from deepclustering.dataset.classification import (
    Cifar10ClusteringDatasetInterface,
    default_cifar10_img_transform,
)
from deepclustering.manager import ConfigManger
from deepclustering.model import Model
from playground.IIC_VAT.VATIICTrainer import IMSATIICTrainer

DEFAULT_CONFIG = str(Path(__file__).parent / "config.yaml")

config = ConfigManger(DEFAULT_CONFIG_PATH=DEFAULT_CONFIG, verbose=True).config
# create model:
model = Model(
    arch_dict=config["Arch"],
    optim_dict=config["Optim"],
    scheduler_dict=config["Scheduler"],
)

train_loader_A = Cifar10ClusteringDatasetInterface(
    **config["DataLoader"]
).ParallelDataLoader(
    default_cifar10_img_transform["tf1"],
    default_cifar10_img_transform["tf2"],
    default_cifar10_img_transform["tf2"],
    default_cifar10_img_transform["tf2"],
    default_cifar10_img_transform["tf2"],
)
train_loader_B = Cifar10ClusteringDatasetInterface(
    **config["DataLoader"]
).ParallelDataLoader(
    default_cifar10_img_transform["tf1"],
    default_cifar10_img_transform["tf2"],
    default_cifar10_img_transform["tf2"],
    default_cifar10_img_transform["tf2"],
    default_cifar10_img_transform["tf2"],
)
val_loader = Cifar10ClusteringDatasetInterface(
    **config["DataLoader"]
).ParallelDataLoader(default_cifar10_img_transform["tf3"])

trainer = IMSATIICTrainer(
    model=model,
    train_loader_A=train_loader_A,
    train_loader_B=train_loader_B,
    val_loader=val_loader,
    config=config,
    **config["Trainer"]
)
trainer.start_training()
trainer.clean_up()
