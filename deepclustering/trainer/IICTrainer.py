from typing import *

import matplotlib.pyplot as plt
import torch
from deepclustering.model import Model
from torch.utils.data import DataLoader

from .trainer import _Trainer
from .. import ModelMode
from ..loss.IID_losses import IIDLoss
from ..meters import MeterInterface, AverageValueMeter
from ..utils import tqdm_, flatten_dict
from ..utils.classification.assignment_mapping import hungarian_match, flat_acc

plt.ion()


class IICTrainer(_Trainer):
    METER_CONFIG = {'traloss': AverageValueMeter(), 'val_acc': AverageValueMeter()}
    METERINTERFACE = MeterInterface(METER_CONFIG)

    def __init__(self, model: Model, train_loader: DataLoader, val_loader: DataLoader, max_epoch: int = 100,
                 save_dir: str = 'runs/test', checkpoint_path: str = None, device='cpu') -> None:
        super().__init__(model, train_loader, val_loader, max_epoch, save_dir, checkpoint_path, device)

    def start_training(self):

        for epoch in range(self.max_epoch):
            self._train_loop(self.train_loader, epoch)
            with torch.no_grad():
                self._eval_loop(self.val_loader, epoch)
            self.METERINTERFACE.step()

            print(self.METERINTERFACE.summary())

    def _train_loop(self, train_loader: DataLoader, epoch: int, mode: ModelMode = ModelMode.TRAIN, *args, **kwargs):
        self.model.set_mode(mode)
        assert self.model.training
        train_loader = tqdm_(train_loader)
        for batch, image_labels in enumerate(train_loader):
            images, _ = list(zip(*image_labels))
            tf1_images = torch.cat([images[0] for _ in range(images.__len__() - 1)], dim=0).to(self.device)
            tf2_images = torch.cat(images[1:], dim=0).to(self.device)
            assert tf1_images.shape == tf2_images.shape
            self.model.zero_grad()
            tf1_pred_simplex = self.model(tf1_images)
            tf2_pred_simplex = self.model(tf2_images)
            assert tf1_pred_simplex.__len__() == tf2_pred_simplex.__len__()
            iicloss: List[torch.Tensor] = []
            for n_subhead in range(tf2_pred_simplex.__len__()):
                iicloss.append(IIDLoss()(tf1_pred_simplex[n_subhead], tf2_pred_simplex[n_subhead])[0])
            loss: torch.Tensor = sum(iicloss) / len(iicloss)  # type: ignore
            loss.backward()
            self.model.step()
            self.METERINTERFACE.traloss.add(-loss.item())
            report_dict = flatten_dict({'MI': self.METERINTERFACE.traloss.summary()}, sep=' ')
            train_loader.set_postfix(report_dict)

    def _eval_loop(self, val_loader: DataLoader, epoch: int, mode: ModelMode = ModelMode.EVAL, *args, **kwargs):
        self.model.set_mode(mode)
        assert not self.model.training
        val_loader_ = tqdm_(val_loader)
        preds = torch.zeros(self.model.arch_dict['num_sub_heads'], val_loader.dataset.__len__(),
                            dtype=torch.long, device=self.device)
        target = torch.zeros(val_loader.dataset.__len__(), dtype=torch.long, device=self.device)
        slice_done = 0
        for batch, image_labels in enumerate(val_loader_):
            images, gt = list(zip(*image_labels))
            images, gt = images[0].to(self.device), gt[0].to(self.device)
            _pred = self.model(images)
            bslicer = slice(slice_done, slice_done + gt.shape[0])
            for subhead in range(preds.__len__()):
                preds[subhead][bslicer] = _pred[subhead].max(1)[1]
            target[bslicer] = gt
            slice_done += gt.shape[0]
        for subhead in range(preds.__len__()):
            reorder_pred, remap = hungarian_match(flat_preds=preds[subhead], flat_targets=target, preds_k=10,
                                                  targets_k=10)
            acc = flat_acc(reorder_pred, target)
            self.METERINTERFACE.val_acc.add(acc)

    @property
    def state_dict(self):
        pass
