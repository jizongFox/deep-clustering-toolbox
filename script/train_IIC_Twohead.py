from pprint import pprint
from typing import Dict, Any

from deepclustering.model import Model
from deepclustering.trainer.IICMultiheadTrainer import IICMultiHeadTrainer
from deepclustering.utils import yaml_parser, yaml_load, dict_merge

DEFAULT_CONFIG = '../config/IICClusterMultiHead_CIFAR.yaml'

parsed_args: Dict[str, Any] = yaml_parser(verbose=True)
default_config = yaml_load(parsed_args.get('Config', DEFAULT_CONFIG), verbose=False)
merged_config = dict_merge(default_config, parsed_args, re=True)
print('Merged config:')
pprint(merged_config)


def get_dataloader(config: dict):
    if config.get('Config', DEFAULT_CONFIG).split('_')[-1].lower() == 'cifar.yaml':
        from deepclustering.dataset import default_cifar10_img_transform as img_transforms, \
            Cifar10ClusteringDatasetInterface as DatasetInterface
        train_split_partition = ['train', 'val']
        val_split_partition = ['train', 'val']

    elif config.get('Config', DEFAULT_CONFIG).split('_')[-1].lower() == 'mnist.yaml':
        from deepclustering.dataset import default_mnist_img_transform as img_transforms, \
            MNISTClusteringDatasetInterface as DatasetInterface
        train_split_partition = ['train', 'val']
        val_split_partition = ['train', 'val']
    elif config.get('Config', DEFAULT_CONFIG).split('_')[-1].lower() == 'stl10.yaml':
        from deepclustering.dataset import default_stl10_img_transform as img_transforms, \
            STL10DatasetInterface as DatasetInterface
        train_split_partition = ['train', 'test', 'train+unlabeled']
        val_split_partition = ['train', 'test']
    else:
        raise NotImplementedError(config.get('Config', DEFAULT_CONFIG).split('_')[-1].lower())
    train_loader_A = DatasetInterface(split_partitions=train_split_partition,
                                      **merged_config['DataLoader']).ParallelDataLoader(
        img_transforms['tf1'],
        img_transforms['tf2'],
        img_transforms['tf2'],
        img_transforms['tf2'],
        img_transforms['tf2'],
    )
    train_loader_B = DatasetInterface(split_partitions=train_split_partition,
                                      **merged_config['DataLoader']).ParallelDataLoader(
        img_transforms['tf1'],
        img_transforms['tf2'],
        img_transforms['tf2'],
        img_transforms['tf2'],
        img_transforms['tf2'],
    )
    val_loader = DatasetInterface(split_partitions=val_split_partition,
                                  **merged_config['DataLoader']).ParallelDataLoader(
        img_transforms['tf3'],
    )
    return train_loader_A, train_loader_B, val_loader


train_loader_A, train_loader_B, val_loader = get_dataloader(merged_config)

# create model:
model = Model(
    arch_dict=merged_config['Arch'],
    optim_dict=merged_config['Optim'],
    scheduler_dict=merged_config['Scheduler']
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
