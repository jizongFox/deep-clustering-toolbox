from abc import ABC, abstractmethod

from pathlib2 import Path
from torch.utils.data import DataLoader

from ..model import Model


class _Trainer(ABC):
    """
    Abstract class for a general trainer, which has _train_loop, _eval_loop,load_state, state_dict, and save_checkpoint
    functions. All other trainers are the subclasses of this class.
    """

    def __init__(self, model: Model, train_loader: DataLoader, val_loader: DataLoader, max_epoch: int = 100,
                 save_dir: str = 'runs/test', checkpoint_path: str = None) -> None:
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir: Path = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.checkpoint = checkpoint_path
        self.max_epoch = int(max_epoch)
        self._start_epoch = 0  # whether 0 or loaded from the checkpoint.

    def start_training(self):
        for epoch in range(self._start_epoch, self.max_epoch):
            self._train_loop()
            self._eval_loop()
            self.save_checkpoint()

    def to(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _train_loop(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _eval_loop(self, *args, **kwargs):
        raise NotImplementedError

    @property
    @abstractmethod
    def state_dict(self):
        raise NotImplementedError

    def save_checkpoint(self):
        raise NotImplementedError
