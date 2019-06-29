from typing import List

from deepclustering.model import Model
from deepclustering.trainer.Trainer import _Trainer
from torch.utils.data import DataLoader


class subspaceTrainer(_Trainer):
    def __init__(
        self,
        model: Model,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
        val_loader: DataLoader,
        max_epoch: int = 100,
        save_dir: str = "subspace",
        checkpoint_path: str = None,
        device="cpu",
        config: dict = None,
    ) -> None:
        super().__init__(
            model,
            None,
            val_loader,
            max_epoch,
            save_dir,
            checkpoint_path,
            device,
            config,
        )
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        assert self.train_loader == None

    def __init_meters__(self) -> List[str]:
        return [""]

    def start_training(self):
        pass
