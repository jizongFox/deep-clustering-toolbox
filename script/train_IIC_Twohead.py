from pprint import pprint
from typing import Dict, Any

from deepclustering.dataset import default_cifar10_img_transform, STL10DatasetInterface
from deepclustering.model import Model
from deepclustering.trainer.IICMultiheadTrainer import IICMultiHeadTrainer
from deepclustering.utils import yaml_parser, yaml_load, dict_merge

DEFAULT_CONFIG = '../config/IICClusterMultiHead.yaml'

parsed_args: Dict[str, Any] = yaml_parser(verbose=True)
default_config = yaml_load(parsed_args.get('Config', DEFAULT_CONFIG), verbose=False)
merged_config = dict_merge(default_config, parsed_args, re=True)
print('Merged config:')
pprint(merged_config)

# create model:
model = Model(
    arch_dict=merged_config['Arch'],
    optim_dict=merged_config['Optim'],
    scheduler_dict=merged_config['Scheduler']
)

train_loader_A = STL10DatasetInterface(split_partitions=['train+unlabeled', 'test'],
                                       **merged_config['DataLoader']).ParallelDataLoader(
    default_cifar10_img_transform['tf1'],
    default_cifar10_img_transform['tf2'],
    default_cifar10_img_transform['tf2'],
    default_cifar10_img_transform['tf2'],
    default_cifar10_img_transform['tf2'],
)
train_loader_B = STL10DatasetInterface(split_partitions=['train', 'test'],
                                       **merged_config['DataLoader']).ParallelDataLoader(
    default_cifar10_img_transform['tf1'],
    default_cifar10_img_transform['tf2'],
    default_cifar10_img_transform['tf2'],
    default_cifar10_img_transform['tf2'],
    default_cifar10_img_transform['tf2'],
)
val_loader = STL10DatasetInterface(split_partitions=['train', 'test'],
                                   **merged_config['DataLoader']).ParallelDataLoader(
    default_cifar10_img_transform['tf3'],
)

trainer = IICMultiHeadTrainer(
    model=model,
    train_loader_A=train_loader_A,
    train_loader_B=train_loader_B,
    val_loader=val_loader,
    config=merged_config,
    **merged_config['Trainer']
)
trainer.start_training()
trainer.clean_up()
