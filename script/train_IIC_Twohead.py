from typing import Tuple

from deepclustering.manager import ConfigManger
from deepclustering.model import Model, to_Apex
from deepclustering.trainer.IICMultiheadTrainer import IICMultiHeadTrainer
from torch.utils.data import DataLoader


def get_dataloader(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    get dataloader for IIC project
    :param config:
    :return:
    """
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
        img_transforms['tf2'],
    )
    train_loader_B = DatasetInterface(split_partitions=train_split_partition,
                                      **merged_config['DataLoader']).ParallelDataLoader(
        img_transforms['tf1'],
        img_transforms['tf2'],
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


DEFAULT_CONFIG = '../config/IICClusterMultiHead_MNIST.yaml'

merged_config = ConfigManger(DEFAULT_CONFIG_PATH=DEFAULT_CONFIG, verbose=True, integrality_check=True).config

train_loader_A, train_loader_B, val_loader = get_dataloader(merged_config)

# create model:
model = Model(
    arch_dict=merged_config['Arch'],
    optim_dict=merged_config['Optim'],
    scheduler_dict=merged_config['Scheduler'],
)
model = to_Apex(model, opt_level=None, verbosity=1)

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
