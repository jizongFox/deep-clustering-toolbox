import warnings
from pprint import pprint
from typing import Dict, Any

from deepclustering.dataset.classification import (
    MNISTClusteringDatasetInterface,
    default_mnist_img_transform,
)
from deepclustering.model import Model
from deepclustering.trainer.IMSATTrainer import IMSATTrainer
from deepclustering.utils import yaml_parser, yaml_load, dict_merge, fix_all_seed

warnings.filterwarnings("ignore")

fix_all_seed(2)

DEFAULT_CONFIG = "../config/IMSAT.yaml"

parsed_args: Dict[str, Any] = yaml_parser(verbose=True)
default_config = yaml_load(parsed_args.get("Config", DEFAULT_CONFIG), verbose=False)
merged_config = dict_merge(default_config, parsed_args, re=True)
print("Merged config:")
pprint(merged_config)

# create model:
model = Model(
    arch_dict=merged_config["Arch"],
    optim_dict=merged_config["Optim"],
    scheduler_dict=merged_config["Scheduler"],
)

train_loader_A = MNISTClusteringDatasetInterface(
    **merged_config["DataLoader"]
).ParallelDataLoader(
    default_mnist_img_transform["tf1"],
    default_mnist_img_transform["tf2"],
    default_mnist_img_transform["tf2"],
    default_mnist_img_transform["tf2"],
    default_mnist_img_transform["tf2"],
)

val_loader = MNISTClusteringDatasetInterface(
    **merged_config["DataLoader"]
).ParallelDataLoader(default_mnist_img_transform["tf3"])

trainer = IMSATTrainer(
    model=model,
    train_loader=train_loader_A,
    val_loader=val_loader,
    config=merged_config,
    **merged_config["Trainer"]
)
trainer.start_training()
trainer.clean_up()
