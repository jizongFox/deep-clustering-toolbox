from typing import List

import matplotlib
import numpy as np
import torch
from pathlib2 import Path
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepclustering.model import Model

matplotlib.use("tkagg")

from .Trainer import _Trainer
from .. import ModelMode
from ..loss.IMSAT_loss import Perturbation_Loss, MultualInformaton_IMSAT
from ..meters import AverageValueMeter, MeterInterface
from ..utils import tqdm_, simplex
from ..utils.VAT import VATLoss
from ..utils.classification.assignment_mapping import hungarian_match, flat_acc


class IMSATTrainer(_Trainer):
    """
    Trainer specific for IMSAT paper
    """

    def __init__(
        self,
        model: Model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epoch: int = 1,
        save_dir: str = "./runs/IMSAT",
        checkpoint_path: str = None,
        device="cpu",
        config: dict = None,
    ) -> None:
        super().__init__(
            model,
            train_loader,
            val_loader,
            max_epoch,
            save_dir,
            checkpoint_path,
            device,
            config,
        )
        self.SAT_criterion = Perturbation_Loss()
        self.MI_criterion = MultualInformaton_IMSAT()
        nearest_dict = np.loadtxt(
            Path(__file__).parents[1] / "dataset/classification/10th_neighbor.txt"
        )
        self.nearest_dict = torch.from_numpy(nearest_dict).float().to(self.device)

    def __init_meters__(self) -> List[str]:
        METER_CONFIG = {
            "train_sat_loss": AverageValueMeter(),
            "train_mi_loss": AverageValueMeter(),
            "val_acc": AverageValueMeter(),
        }
        self.METERINTERFACE = MeterInterface(METER_CONFIG)
        return ["train_sat_loss_mean", "train_mi_loss_mean", "val_acc_mean"]

    @property
    def _training_report_dict(self):
        report_dict = {
            "sat": self.METERINTERFACE["train_sat_loss"].summary()["mean"],
            "mi": self.METERINTERFACE["train_mi_loss"].summary()["mean"],
        }
        return report_dict

    @property
    def _eval_report_dict(self):
        report_dict = {"val_acc": self.METERINTERFACE.val_acc.summary()["mean"]}
        return report_dict

    def _train_loop(
        self, train_loader, epoch, mode: ModelMode = ModelMode.TRAIN, **kwargs
    ):
        self.model.set_mode(mode)
        assert (
            self.model.training
        ), f"Model should be in train() model, given {self.model.training}."
        train_loader_: tqdm = tqdm_(train_loader)  # reinitilize the train_loader
        train_loader_.set_description(f"Training epoch: {epoch}")
        for batch, image_labels in enumerate(train_loader_):
            images, _, (index, *_) = list(zip(*image_labels))
            # print(f"used time for dataloading:{time.time() - time_before}")
            tf1_images = torch.cat(
                [images[0] for _ in range(images.__len__() - 1)], dim=0
            ).to(self.device)
            tf2_images = torch.cat(images[1:], dim=0).to(self.device).to(self.device)
            index = torch.cat([index for _ in range(images.__len__() - 1)], dim=0)

            tf1_images = tf1_images.view(tf1_images.shape[0], -1)
            tf2_images = tf2_images.view(tf2_images.shape[0], -1)

            assert tf1_images.shape == tf2_images.shape
            tf1_pred_logit = self.model.torchnet(tf1_images)
            tf2_pred_logit = self.model.torchnet(tf2_images)
            assert (
                not simplex(tf1_pred_logit)
                and tf1_pred_logit.shape == tf2_pred_logit.shape
            )
            VAT_generator = VATLoss(eps=self.nearest_dict[index])
            vat_loss, adv_tf1_images, _ = VAT_generator(self.model.torchnet, tf1_images)

            sat_loss = self.SAT_criterion(tf1_pred_logit.detach(), tf2_pred_logit)

            ml_loss, *_ = self.MI_criterion(tf1_pred_logit)
            # sat_loss = torch.Tensor([0]).cuda()
            batch_loss: torch.Tensor = vat_loss + sat_loss - 0.1 * ml_loss
            # batch_loss: torch.Tensor = - 0.1 * ml_loss

            self.METERINTERFACE["train_sat_loss"].add(vat_loss.item())
            self.METERINTERFACE["train_mi_loss"].add(ml_loss.item())
            self.model.zero_grad()
            batch_loss.backward()
            self.model.step()
            report_dict = self._training_report_dict
            train_loader_.set_postfix(report_dict)

    def _eval_loop(
        self,
        val_loader: DataLoader,
        epoch: int,
        mode: ModelMode = ModelMode.EVAL,
        **kwargs,
    ) -> float:
        self.model.set_mode(mode)
        assert (
            not self.model.training
        ), f"Model should be in eval model in _eval_loop, given {self.model.training}."
        val_loader_: tqdm = tqdm_(val_loader)
        preds = torch.zeros(
            val_loader.dataset.__len__(), dtype=torch.long, device=self.device
        )
        target = torch.zeros(
            val_loader.dataset.__len__(), dtype=torch.long, device=self.device
        )
        slice_done = 0
        subhead_accs = []
        val_loader_.set_description(f"Validating epoch: {epoch}")
        for batch, image_labels in enumerate(val_loader_):
            images, gt, _ = list(zip(*image_labels))
            images, gt = images[0].to(self.device), gt[0].to(self.device)
            _pred = F.softmax(self.model.torchnet(images.view(images.size(0), -1)), 1)
            assert simplex(_pred)
            bSlicer = slice(slice_done, slice_done + images.shape[0])
            preds[bSlicer] = _pred.max(1)[1]
            target[bSlicer] = gt
            slice_done += gt.shape[0]
        assert slice_done == val_loader.dataset.__len__(), "Slice not completed."

        reorder_pred, remap = hungarian_match(
            flat_preds=preds, flat_targets=target, preds_k=10, targets_k=10
        )
        _acc = flat_acc(reorder_pred, target)
        subhead_accs.append(_acc)
        # record average acc
        self.METERINTERFACE.val_acc.add(_acc)

        # record best acc
        report_dict = self._eval_report_dict
        report_dict_str = ", ".join([f"{k}:{v:.3f}" for k, v in report_dict.items()])
        print(f"Validating epoch: {epoch} : {report_dict_str}")
        return self.METERINTERFACE.val_acc.summary()["mean"]
