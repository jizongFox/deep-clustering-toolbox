from typing import *
from torch.utils.data import DataLoader

from deepclustering import ModelMode
from deepclustering.decorator import lazy_load_checkpoint
from deepclustering.loss import KL_div
from deepclustering.model import Model
from deepclustering.trainer import _Trainer
from deepclustering.meters import AverageValueMeter, SliceDiceMeter, BatchDiceMeter


class SemiTrainer(_Trainer):

    @lazy_load_checkpoint
    def __init__(self, model: Model, labeled_loader: DataLoader, unlabeled_loader: DataLoader, val_loader: DataLoader,
                 max_epoch: int = 100, save_dir: str = "base", checkpoint_path: str = None, device="cpu",
                 config: dict = None, **kwargs) -> None:
        super().__init__(model, None, val_loader, max_epoch, save_dir, checkpoint_path, device, config, **kwargs)
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        assert self.train_loader == None
        self.kl_criterion = KL_div()

    def __init_meters__(self) -> List[Union[str, List[str]]]:
        meter_config= {"trloss":AverageValueMeter(),
                       "trdice":SliceDiceMeter()}


    def _train_loop(
            self, labeled_loader: DataLoader = None, unlabeled_loader: DataLoader = None, epoch: int = 0,
            mode=ModelMode.TRAIN, *args, **kwargs
    ):
        self.model.set_mode(mode)
        for batch_num, (((lab_img, lab_gt), lab_path), ((unlab_img, _), unlab_path)) in enumerate(
                zip(labeled_loader, unlabeled_loader)):
            print()
