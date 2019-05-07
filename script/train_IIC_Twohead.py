from deepclustering.model import Model
from deepclustering.dataset.classification import Cifar10ClusteringDataloaders, default_cifar10_img_transform
from deepclustering.trainer.IICMultiheadTrainer import IICMultiHeadTrainer
from deepclustering.utils import yaml_parser, yaml_load, dict_merge

from typing import Dict, Any
from pprint import pprint

DEFAULT_CONFIG = 'config/IICClusterMultiHead.yaml'

parsed_args: Dict[str, Any] = yaml_parser(verbose=True)
default_config = yaml_load(parsed_args.get('Config', DEFAULT_CONFIG), True)
merged_config = dict_merge(default_config, parsed_args, re=True)
print('Merged config:')
pprint(merged_config)

# create model:
model = Model(
    arch_dict=merged_config['Arch'],
    optim_dict=merged_config['Optim'],
    scheduler_dict=merged_config['Scheduler']
)
train_loader_A = Cifar10ClusteringDataloaders(**merged_config['DataLoader']).creat_CombineDataLoader(
    default_cifar10_img_transform['tf1'],
    default_cifar10_img_transform['tf2'],
    default_cifar10_img_transform['tf2'],
    default_cifar10_img_transform['tf2'],
    default_cifar10_img_transform['tf2'],
)
train_loader_B = Cifar10ClusteringDataloaders(**merged_config['DataLoader']).creat_CombineDataLoader(
    default_cifar10_img_transform['tf1'],
    default_cifar10_img_transform['tf2'],
    default_cifar10_img_transform['tf2'],
    default_cifar10_img_transform['tf2'],
    default_cifar10_img_transform['tf2'],
)
val_loader = Cifar10ClusteringDataloaders(**merged_config['DataLoader']).creat_CombineDataLoader(
    default_cifar10_img_transform['tf3']
)

trainer = IICMultiHeadTrainer(
    model=model,
    train_loader_A=train_loader_A,
    train_loader_B=train_loader_B,
    val_loader=val_loader,
    **merged_config['Trainer']
)
trainer.start_training()
