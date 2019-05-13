from abc import ABC, abstractmethod
from copy import deepcopy as dcopy

import torch
import yaml
from pathlib2 import Path
from torch.utils.data import DataLoader

from ..model import Model


class _Trainer(ABC):
    """
    Abstract class for a general trainer, which has _train_loop, _eval_loop,load_state, state_dict, and save_checkpoint
    functions. All other trainers are the subclasses of this class.
    """
    METER_CONFIG = None
    METERINTERFACE = None

    def __init__(self, model: Model, train_loader: DataLoader, val_loader: DataLoader, max_epoch: int = 100,
                 save_dir: str = './runs/test', checkpoint_path: str = None, device='cpu', config: dict = None) -> None:
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir: Path = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        (self.save_dir / 'meters').mkdir(exist_ok=True, parents=True)
        self.checkpoint = checkpoint_path
        self.max_epoch = int(max_epoch)
        self.best_score: float = -1
        self._start_epoch = 0  # whether 0 or loaded from the checkpoint.
        self.device = torch.device(device)
        if checkpoint_path:
            assert Path(checkpoint_path).exists() and Path(checkpoint_path).is_dir()
            state_dict = torch.load(str(Path(checkpoint_path) / 'last.pth'), map_location=torch.device('cpu'))
            self.load_checkpoint(state_dict)

        if config:
            # save config file to save_dir
            self.config = dcopy(config)
            try:
                self.config.pop('Config')
            except KeyError:
                pass
            with open(self.save_dir / 'config.yaml', 'w') as outfile:
                yaml.dump(self.config, outfile, default_flow_style=False)
        self.model.to(self.device)

    def start_training(self):
        for epoch in range(self._start_epoch + 1, self.max_epoch):
            self._train_loop()
            self._eval_loop()
            self.save_checkpoint()

    def to(self, device):
        self.model.to(device=device)

    @abstractmethod
    def _train_loop(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _eval_loop(self, *args, **kwargs) -> float:
        """
        return the
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def state_dict(self):
        raise NotImplementedError

    def save_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    def load_checkpoint(self, *args, **kwargs):
        raise NotImplementedError
