from typing import List

import numpy as np
import torch
from pathlib2 import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepclustering import ModelMode
from deepclustering import PROJECT_PATH
from deepclustering.loss.IMSAT_loss import Perturbation_Loss, MultualInformaton_IMSAT
from deepclustering.meters import AverageValueMeter, MeterInterface
from deepclustering.model import Model
from deepclustering.trainer.Trainer import _Trainer
from deepclustering.utils import tqdm_, simplex, assert_list
from deepclustering.utils.VAT import VATLoss_Multihead
from deepclustering.utils.classification.assignment_mapping import (
    hungarian_match,
    flat_acc,
)


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
            Path(PROJECT_PATH) / "playground/IMSAT/10th_neighbor.txt"
        )
        self.nearest_dict = torch.from_numpy(nearest_dict).float().to(self.device)

    def __init_meters__(self) -> List[str]:
        METER_CONFIG = {
            "train_adv_loss": AverageValueMeter(),
            "train_sat_loss": AverageValueMeter(),
            "train_mi_loss": AverageValueMeter(),
            "val_avg_acc": AverageValueMeter(),
            "val_best_acc": AverageValueMeter(),
            "val_worst_acc": AverageValueMeter(),
        }
        self.METERINTERFACE = MeterInterface(METER_CONFIG)
        return [
            "train_mi_loss_mean",
            "train_sat_loss_mean",
            "train_adv_loss_mean",
            ["val_avg_acc_mean", "val_best_acc_mean", "val_worst_acc_mean"],
        ]

    @property
    def _training_report_dict(self):
        report_dict = {
            "mi": self.METERINTERFACE["train_mi_loss"].summary()["mean"],
            "sat": self.METERINTERFACE["train_sat_loss"].summary()["mean"],
            "adv": self.METERINTERFACE["train_adv_loss"].summary()["mean"],
        }
        return report_dict

    @property
    def _eval_report_dict(self):
        report_dict = {
            "val_avg_acc": self.METERINTERFACE.val_avg_acc.summary()["mean"],
            "val_best_acc": self.METERINTERFACE.val_best_acc.summary()["mean"],
            "val_worst_acc": self.METERINTERFACE.val_worst_acc.summary()["mean"],
        }
        return report_dict

    def _train_loop(
        self, train_loader=None, epoch=0, mode: ModelMode = ModelMode.TRAIN, **kwargs
    ):
        self.model.set_mode(mode)
        assert (
            self.model.training
        ), f"Model should be in train() model, given {self.model.training}."
        train_loader_: tqdm = tqdm_(train_loader)
        train_loader_.set_description(f"Training epoch: {epoch}")
        for batch, image_labels in enumerate(train_loader_):
            images, _, (index, *_) = list(zip(*image_labels))
            tf1_images = torch.cat(
                [images[0] for _ in range(images.__len__() - 1)], dim=0
            ).to(self.device)
            tf2_images = torch.cat(images[1:], dim=0).to(self.device).to(self.device)
            index = torch.cat([index for _ in range(images.__len__() - 1)], dim=0)

            assert tf1_images.shape == tf2_images.shape
            tf1_pred_logit = self.model.torchnet(tf1_images)
            tf2_pred_logit = self.model.torchnet(tf2_images)
            assert (
                assert_list(simplex, tf1_pred_logit)
                and tf1_pred_logit[0].shape == tf2_pred_logit[0].shape
            )

            sat_losses = []
            ml_losses = []
            for subhead_num, (tf1_pred, tf2_pred) in enumerate(
                zip(tf1_pred_logit, tf2_pred_logit)
            ):
                sat_loss = self.SAT_criterion(tf2_pred, tf1_pred.detach())
                ml_loss, *_ = self.MI_criterion(tf1_pred)
                # sat_losses.append(sat_loss)
                ml_losses.append(ml_loss)
            ml_losses = sum(ml_losses) / len(ml_losses)
            # sat_losses = sum(sat_losses) / len(sat_losses)

            # VAT_generator = VATLoss_Multihead(eps=self.nearest_dict[index])
            VAT_generator = VATLoss_Multihead(eps=10)
            vat_loss, adv_tf1_images, _ = VAT_generator(self.model.torchnet, tf1_images)

            batch_loss: torch.Tensor = vat_loss - 0.1 * ml_losses

            # self.METERINTERFACE["train_sat_loss"].add(sat_losses.item())
            self.METERINTERFACE["train_mi_loss"].add(ml_losses.item())
            self.METERINTERFACE["train_adv_loss"].add(vat_loss.item())
            self.model.zero_grad()
            batch_loss.backward()
            self.model.step()
            report_dict = self._training_report_dict
            train_loader_.set_postfix(report_dict)

    def _eval_loop(
        self,
        val_loader: DataLoader = None,
        epoch: int = 0,
        mode: ModelMode = ModelMode.EVAL,
        **kwargs,
    ) -> float:
        self.model.set_mode(mode)
        assert (
            not self.model.training
        ), f"Model should be in eval model in _eval_loop, given {self.model.training}."
        val_loader_: tqdm = tqdm_(val_loader)
        preds = torch.zeros(
            self.model.arch_dict["num_sub_heads"],
            val_loader.dataset.__len__(),
            dtype=torch.long,
            device=self.device,
        )
        target = torch.zeros(
            val_loader.dataset.__len__(), dtype=torch.long, device=self.device
        )
        slice_done = 0
        subhead_accs = []
        val_loader_.set_description(f"Validating epoch: {epoch}")
        for batch, image_labels in enumerate(val_loader_):
            images, gt, *_ = list(zip(*image_labels))
            images, gt = images[0].to(self.device), gt[0].to(self.device)
            _pred = self.model.torchnet(images)
            assert (
                assert_list(simplex, _pred)
                and _pred.__len__() == self.model.arch_dict["num_sub_heads"]
            )
            bSlicer = slice(slice_done, slice_done + images.shape[0])
            for subhead in range(self.model.arch_dict["num_sub_heads"]):
                preds[subhead][bSlicer] = _pred[subhead].max(1)[1]
            target[bSlicer] = gt
            slice_done += gt.shape[0]
        assert slice_done == val_loader.dataset.__len__(), "Slice not completed."

        for subhead in range(self.model.arch_dict["num_sub_heads"]):
            reorder_pred, remap = hungarian_match(
                flat_preds=preds[subhead],
                flat_targets=target,
                preds_k=self.model.arch_dict["output_k_B"],
                targets_k=self.model.arch_dict["output_k_B"],
            )
            _acc = flat_acc(reorder_pred, target)
            subhead_accs.append(_acc)
            # record average acc
            self.METERINTERFACE.val_avg_acc.add(_acc)
        # record best acc
        self.METERINTERFACE.val_best_acc.add(max(subhead_accs))
        self.METERINTERFACE.val_worst_acc.add(min(subhead_accs))
        report_dict = self._eval_report_dict

        report_dict_str = ", ".join([f"{k}:{v:.3f}" for k, v in report_dict.items()])
        print(f"Validating epoch: {epoch} : {report_dict_str}")
        return self.METERINTERFACE.val_best_acc.summary()["mean"]
