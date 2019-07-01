#
#
#   This file is to implement the mixup scheme to MI(f(x),f(t(x))).
#   The function would be original MI + MI(f(x1),f(x1+x2))+ MI(f(x2),f(x1+x2))
#
from pprint import pprint
from typing import *

import torch
from torch.utils.data import DataLoader

from deepclustering.model import Model
from deepclustering.trainer.IICMultiheadTrainer import IICMultiHeadTrainer
from deepclustering.utils import yaml_parser, dict_merge, yaml_load


class MixUpTrainer(IICMultiHeadTrainer):
    def __init__(
        self,
        model: Model,
        train_loader_A: DataLoader,
        train_loader_B: DataLoader,
        val_loader: DataLoader,
        max_epoch: int = 100,
        save_dir: str = "./runs/IICMultiHead",
        checkpoint_path: str = None,
        device="cpu",
        head_control_params: dict = {},
        mixup_weight: float = 0,
        use_sobel: bool = True,
        config: dict = None,
    ) -> None:
        super().__init__(
            model,
            train_loader_A,
            train_loader_B,
            val_loader,
            max_epoch,
            save_dir,
            checkpoint_path,
            device,
            head_control_params,
            use_sobel,
            config,
        )
        self.mixup_weight = float(mixup_weight)

    def _trainer_specific_loss(self, tf1_images, tf2_images, head_name):
        """
        training loss for MixUp training
        :param tf1_images: images preprocessed with tf1 transformation (basic)
        :param tf2_images: images preprocessed with tf2 transformation (random)
        :param head_name: the head used in the two head setting (A or B, with A being overclustering head)
        :return: loss
        """
        orignal_loss = super(MixUpTrainer, self)._trainer_specific_loss(
            tf1_images, tf2_images, head_name
        )
        mixup_loss = 0
        if self.mixup_weight > 0:
            bn, *shapes = tf1_images.shape
            alpha = torch.rand(bn).view(bn, 1, 1, 1).repeat(1, *shapes).to(self.device)
            mix_images = alpha * tf1_images + (1 - alpha) * tf2_images
            # create a list of alpha that is the mixing coefficient
            mixup_loss = super(MixUpTrainer, self)._trainer_specific_loss(
                tf1_images, mix_images, head_name
            ) + super(MixUpTrainer, self)._trainer_specific_loss(
                tf2_images, mix_images, head_name
            )
        return orignal_loss + self.mixup_weight * mixup_loss


DEFAULT_CONFIG = "IICClusterMultiHead_CIFAR.yaml"

parsed_args: Dict[str, Any] = yaml_parser(verbose=True)
default_config = yaml_load(parsed_args.get("Config", DEFAULT_CONFIG), verbose=False)
merged_config = dict_merge(default_config, parsed_args, re=True)
print("Merged config:")
pprint(merged_config)


def get_dataloader(config: dict):
    if config.get("Config", DEFAULT_CONFIG).split("_")[-1].lower() == "cifar.yaml":
        from deepclustering.dataset import (
            default_cifar10_img_transform as img_transforms,
            Cifar10ClusteringDatasetInterface as DatasetInterface,
        )

        train_split_partition = ["train", "val"]
        val_split_partition = ["train", "val"]

    elif config.get("Config", DEFAULT_CONFIG).split("_")[-1].lower() == "mnist.yaml":
        from deepclustering.dataset import (
            default_mnist_img_transform as img_transforms,
            MNISTClusteringDatasetInterface as DatasetInterface,
        )

        train_split_partition = ["train", "val"]
        val_split_partition = ["train", "val"]
    elif config.get("Config", DEFAULT_CONFIG).split("_")[-1].lower() == "stl10.yaml":
        from deepclustering.dataset import (
            default_stl10_img_transform as img_transforms,
            STL10DatasetInterface as DatasetInterface,
        )

        train_split_partition = ["train", "test", "train+unlabeled"]
        val_split_partition = ["train", "test"]
    else:
        raise NotImplementedError(
            config.get("Config", DEFAULT_CONFIG).split("_")[-1].lower()
        )
    train_loader_A = DatasetInterface(
        split_partitions=train_split_partition, **merged_config["DataLoader"]
    ).ParallelDataLoader(
        img_transforms["tf1"],
        img_transforms["tf2"],
        img_transforms["tf2"],
        img_transforms["tf2"],
        img_transforms["tf2"],
    )
    train_loader_B = DatasetInterface(
        split_partitions=train_split_partition, **merged_config["DataLoader"]
    ).ParallelDataLoader(
        img_transforms["tf1"],
        img_transforms["tf2"],
        img_transforms["tf2"],
        img_transforms["tf2"],
        img_transforms["tf2"],
    )
    val_loader = DatasetInterface(
        split_partitions=val_split_partition, **merged_config["DataLoader"]
    ).ParallelDataLoader(img_transforms["tf3"])
    return train_loader_A, train_loader_B, val_loader


train_loader_A, train_loader_B, val_loader = get_dataloader(merged_config)

# create model:
model = Model(
    arch_dict=merged_config.get("Arch"),
    optim_dict=merged_config.get("Optim"),
    scheduler_dict=merged_config.get("Scheduler"),
)

trainer = MixUpTrainer(
    model=model,
    train_loader_A=train_loader_A,
    train_loader_B=train_loader_B,
    val_loader=val_loader,
    config=merged_config,
    **merged_config.get("Trainer")
)
trainer.start_training()
trainer.clean_up()
