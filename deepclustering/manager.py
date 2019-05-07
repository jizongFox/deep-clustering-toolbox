from pprint import pprint
from typing import Dict, Any

from deepclustering.dataset import Cifar10ClusteringDataloaders, default_cifar10_img_transform
from deepclustering.model import Model
from deepclustering.utils import yaml_parser, yaml_load, dict_merge
from deepclustering.trainer.IICTrainer import IICTrainer
from functools import partial


class _TrainerManger(object):
    DEFAULT_CONFIG = ""

    def __init__(self) -> None:
        self.parsed_args: Dict[str, Any] = yaml_parser(True)
        self.default_config = yaml_load(self.parsed_args.get('Config', self.DEFAULT_CONFIG), True)
        self.merged_config = dict_merge(self.default_config, self.parsed_args, re=True)
        self._check_integrality(self.merged_config)
        print("Merged args:")
        pprint(self.merged_config)

    def create_model(self):
        model = Model(arch_dict=self.merged_config['Arch'], optim_dict=self.merged_config['Optim'],
                      scheduler_dict=self.merged_config['Scheduler'])
        return model

    def create_dataloders(self):
        train_loader = Cifar10ClusteringDataloaders(**self.merged_config['DataLoader']).creat_CombineDataLoader(
            default_cifar10_img_transform['tf1'],
            default_cifar10_img_transform['tf2'],
            default_cifar10_img_transform['tf2'],
        )
        val_loader = Cifar10ClusteringDataloaders().creat_CombineDataLoader(
            default_cifar10_img_transform['tf3']
        )
        return train_loader, val_loader

    def return_trainer(self):
        return partial(IICTrainer, **self.merged_config['Trainer'])

    @staticmethod
    def _check_integrality(merged_dict=Dict[str, Any]):
        assert merged_dict.get('Arch'), f"Merged dict integrity check failed,{merged_dict.keys()}"
        assert merged_dict.get('Optim'), f"Merged dict integrity check failed,{merged_dict.keys()}"
        assert merged_dict.get('Scheduler'), f"Merged dict integrity check failed,{merged_dict.keys()}"
        assert merged_dict.get('Trainer'), f"Merged dict integrity check failed,{merged_dict.keys()}"


if __name__ == '__main__':
    ti = Trainer_Initialization()
    Model = ti.return_Model()
    train_loader, val_loader = ti.return_DataLoders()
