from typing import *

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from deepclustering import ModelMode
from deepclustering.dataloader import DataIter
from deepclustering.decorator import lazy_load_checkpoint
from deepclustering.loss import KL_div
from deepclustering.meters import (
    AverageValueMeter,
    SliceDiceMeter,
    MeterInterface,
    BatchDiceMeter,
)
from deepclustering.model import Model, ZeroGradientBackwardStep
from deepclustering.trainer import _Trainer
from deepclustering.utils import class2one_hot, tqdm_, flatten_dict, nice_dict, one_hot


class SemiSegTrainer(_Trainer):
    @lazy_load_checkpoint
    def __init__(
        self,
        model: Model,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
        val_loader: DataLoader,
        max_epoch: int = 100,
        save_dir: str = "base",
        checkpoint_path: str = None,
        device="cpu",
        config: dict = None,
        max_iter: int = 100,
        axis=(1, 2, 3),
        **kwargs,
    ) -> None:
        self.axis = axis
        super().__init__(
            model,
            None,
            val_loader,
            max_epoch,
            save_dir,
            checkpoint_path,
            device,
            config,
            **kwargs,
        )
        assert self.train_loader is None
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.kl_criterion = KL_div()
        self.max_iter = max_iter

    def __init_meters__(self) -> List[Union[str, List[str]]]:
        meter_config = {
            "lr": AverageValueMeter(),
            "trloss": AverageValueMeter(),
            "trdice": SliceDiceMeter(
                C=self.model.arch_dict["num_classes"], report_axises=self.axis
            ),
            "valloss": AverageValueMeter(),
            "valdice": SliceDiceMeter(
                C=self.model.arch_dict["num_classes"], report_axises=self.axis
            ),
            "valbdice": BatchDiceMeter(
                C=self.model.arch_dict["num_classes"], report_axises=self.axis
            ),
        }
        self.METERINTERFACE = MeterInterface(meter_config)
        return [
            "trloss_mean",
            ["trdice_DSC1", "trdice_DSC2", "trdice_DSC3"],
            "valloss_mean",
            ["valdice_DSC1", "valdice_DSC2", "valdice_DSC3"],
            ["valbdice_DSC1", "valbdice_DSC2", "valbdice_DSC3"],
            "lr_mean",
        ]

    def start_training(self):
        for epoch in range(self._start_epoch, self.max_epoch):
            self._train_loop(
                labeled_loader=self.labeled_loader,
                unlabeled_loader=self.unlabeled_loader,
                epoch=epoch,
            )
            with torch.no_grad():
                current_score = self._eval_loop(self.val_loader, epoch)
            self.METERINTERFACE.step()
            self.model.schedulerStep()
            # save meters and checkpoints
            SUMMARY = self.METERINTERFACE.summary()
            SUMMARY.to_csv(self.save_dir / self.wholemeter_filename)
            self.drawer.draw(SUMMARY)
            self.save_checkpoint(self.state_dict(), epoch, current_score)
        self.writer.close()

    def _train_loop(
        self,
        labeled_loader: DataLoader = None,
        unlabeled_loader: DataLoader = None,
        epoch: int = 0,
        mode=ModelMode.TRAIN,
        *args,
        **kwargs,
    ):
        self.model.set_mode(mode)
        labeled_loader = DataIter(labeled_loader)
        unlabeled_loader = DataIter(unlabeled_loader)
        _max_iter = tqdm_(range(self.max_iter))
        _max_iter.set_description(f"Training Epoch {epoch}")
        self.METERINTERFACE["lr"].add(self.model.get_lr()[0])
        for (
            batch_num,
            ((lab_img, lab_gt), lab_path),
            ((unlab_img, _), unlab_path),
        ) in zip(_max_iter, labeled_loader, unlabeled_loader):
            lab_img, lab_gt = lab_img.to(self.device), lab_gt.to(self.device)
            lab_preds = self.model(lab_img, force_simplex=True)
            sup_loss = self.kl_criterion(
                lab_preds,
                class2one_hot(
                    lab_gt.squeeze(1), C=self.model.arch_dict["num_classes"]
                ).float(),
            )
            reg_loss = self._trainer_specific_loss(unlab_img)
            self.METERINTERFACE["trloss"].add(sup_loss.item())
            self.METERINTERFACE["trdice"].add(lab_preds, lab_gt)

            with ZeroGradientBackwardStep(
                sup_loss + reg_loss, self.model
            ) as total_loss:
                total_loss.backward()
            report_dict = self._training_report_dict
            _max_iter.set_postfix(report_dict)
        print(f"Training Epoch {epoch}: {nice_dict(report_dict)}")
        self.writer.add_scalar_with_tag("train", report_dict, global_step=epoch)

    def _trainer_specific_loss(self, unlab_img: Tensor, **kwargs) -> Tensor:
        return torch.tensor(0, dtype=torch.float32, device=self.device)

    def _eval_loop(
        self,
        val_loader: DataLoader = None,
        epoch: int = 0,
        mode=ModelMode.EVAL,
        *args,
        **kwargs,
    ) -> float:
        self.model.set_mode(mode)
        _val_loader = tqdm_(val_loader)
        _val_loader.set_description(f"Validating Epoch {epoch}")
        for batch_num, ((val_img, val_gt), val_path) in enumerate(_val_loader):
            val_img, val_gt = val_img.to(self.device), val_gt.to(self.device)
            val_preds = self.model(val_img, force_simplex=True)
            val_loss = self.kl_criterion(
                val_preds,
                class2one_hot(
                    val_gt.squeeze(1), C=self.model.arch_dict["num_classes"]
                ).float(),
                disable_assert=True,
            )
            self.METERINTERFACE["valloss"].add(val_loss.item())
            self.METERINTERFACE["valdice"].add(val_preds, val_gt)
            self.METERINTERFACE["valbdice"].add(val_preds, val_gt)
            report_dict = self._eval_report_dict
            _val_loader.set_postfix(report_dict)
        print(f"Validating Epoch {epoch}: {nice_dict(report_dict)}")
        self.writer.add_scalar_with_tag(
            tag="eval", tag_scalar_dict=report_dict, global_step=epoch
        )
        return self.METERINTERFACE["valbdice"].value()[0][0].item()

    @property
    def _training_report_dict(self):
        return flatten_dict(
            {
                "tra_loss": self.METERINTERFACE["trloss"].summary()["mean"],
                "": self.METERINTERFACE["trdice"].summary(),
                "lr": self.METERINTERFACE["lr"].summary()["mean"],
            },
            sep="_",
        )

    @property
    def _eval_report_dict(self):
        return flatten_dict(
            {
                "val_loss": self.METERINTERFACE["valloss"].summary()["mean"],
                "": self.METERINTERFACE["valdice"].summary(),
                "b": self.METERINTERFACE["valbdice"].summary(),
            },
            sep="",
        )


def dice_loss(pred, target):
    assert pred.shape == target.shape
    assert pred.max() <= 1 and pred.min() >= 0
    assert one_hot(target)
    numerator = 2 * torch.sum(pred * target)
    denominator = torch.sum(pred + target)
    return 1 - (numerator + 1e-10) / (denominator + 1e-10)
