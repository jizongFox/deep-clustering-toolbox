import matplotlib.pyplot as plt
import torch
from deepclustering.model import Model
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

from .trainer import _Trainer
from .. import ModelMode
from ..loss.IID_losses import IIDLoss
from ..meters import AggragatedMeter, ListAggregatedMeter, AverageValueMeter
from ..utils import tqdm_

plt.ion()


class IICTrainer(_Trainer):
    METER_NAMES = ['traloss', 'valloss', 'val_acc']
    Meters = edict()
    for k in METER_NAMES:
        Meters[k] = AggragatedMeter()
    MeterInterface = ListAggregatedMeter(names=METER_NAMES, listAggregatedMeter=list(Meters.values()))

    def __init__(self, model: Model, train_loader: DataLoader, val_loader: DataLoader, max_epoch: int = 100,
                 save_dir: str = 'runs/test', checkpoint_path: str = None, device='cpu') -> None:
        super().__init__(model, train_loader, val_loader, max_epoch, save_dir, checkpoint_path, device)

    def start_training(self):

        for epoch in range(self.max_epoch):
            self._train_loop()

    def _train_loop(self, train_loader: DataLoader, epoch: int, mode: ModelMode = ModelMode.TRAIN, *args, **kwargs):
        tralossMeter = AverageValueMeter()
        train_loader = tqdm_(train_loader)
        for batch, images_labels in enumerate(train_loader):
            images, _ = list(zip(*images_labels))
            tf1_images = torch.cat([images[0] for _ in range(images.__len__() - 1)], dim=0).to(self.device)
            tf2_images = torch.cat(images[1:], dim=0).to(self.device)
            assert tf1_images.shape == tf2_images.shape
            self.model.optimizer.zero_grad()
            tf1_pred_logit = self.model(tf1_images)
            tf2_pred_logit = self.model(tf2_images)
            IICloss = []
            for n_subhead in range(tf2_pred_logit.__len__()):
                IICloss.append(IIDLoss()(tf1_pred_logit[n_subhead], tf2_pred_logit[n_subhead])[0])
            loss = sum(IICloss)
            loss.backward()
            self.model.optimizer.step()
            tralossMeter.add(loss.item())
            train_loader.set_postfix(tralossMeter.summary())

    def _eval_loop(self, *args, **kwargs):
        pass

    @property
    def state_dict(self):
        pass
