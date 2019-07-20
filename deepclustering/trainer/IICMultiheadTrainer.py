"""
This is the trainer for IIC multiple-header Clustering
"""

__all__ = ["IICMultiHeadTrainer"]

from collections import OrderedDict
from typing import List, Union

import matplotlib
import torch
from torch.utils.data import DataLoader

from .Trainer import _Trainer
from .. import ModelMode
from ..augment.pil_augment import SobelProcess
from ..loss.IID_losses import IIDLoss
from ..meters import AverageValueMeter, MeterInterface
from ..model import Model, ZeroGradientBackwardStep
from ..utils import tqdm_, simplex, tqdm, dict_filter
from deepclustering.decorator.decorator import TimeBlock
from ..utils.classification.assignment_mapping import flat_acc, hungarian_match

matplotlib.use("agg")


class IICMultiHeadTrainer(_Trainer):
    def __init__(
        self,
        model: Model,
        train_loader_A: DataLoader,
        train_loader_B: DataLoader,
        val_loader: DataLoader,
        max_epoch: int = 100,
        save_dir: str = "./runs/IICMultiHead",
        checkpoint_path: str = None,
        device="cpu",
        head_control_params: dict = {},
        use_sobel: bool = True,
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
        )  # type: ignore
        self.train_loader_A = train_loader_A
        self.train_loader_B = train_loader_B
        assert (
            self.train_loader is None
        ), self.train_loader  # discard the original self.train_loader
        self.head_control_params: OrderedDict = OrderedDict(head_control_params)
        self.criterion = IIDLoss()
        self.criterion.to(self.device)
        self.use_sobel = use_sobel
        if self.use_sobel:
            self.sobel = SobelProcess(include_origin=False)
            self.sobel.to(self.device)

    def __init_meters__(self) -> List[Union[str, List[str]]]:
        METER_CONFIG = {
            "train_head_A": AverageValueMeter(),
            "train_head_B": AverageValueMeter(),
            "val_average_acc": AverageValueMeter(),
            "val_best_acc": AverageValueMeter(),
            "val_worst_acc": AverageValueMeter(),
        }
        self.METERINTERFACE = MeterInterface(METER_CONFIG)
        return [
            "train_head_A_mean",
            "train_head_B_mean",
            ["val_average_acc_mean", "val_best_acc_mean", "val_worst_acc_mean"],
        ]

    @property
    def _training_report_dict(self):
        report_dict = {
            "train_head_A": self.METERINTERFACE["train_head_A"].summary()["mean"],
            "train_head_B": self.METERINTERFACE["train_head_B"].summary()["mean"],
        }
        report_dict = dict_filter(report_dict, lambda k, v: v != 0.0)
        return report_dict

    @property
    def _eval_report_dict(self):
        report_dict = {
            "average_acc": self.METERINTERFACE.val_average_acc.summary()["mean"],
            "best_acc": self.METERINTERFACE.val_best_acc.summary()["mean"],
            "worst_acc": self.METERINTERFACE.val_worst_acc.summary()["mean"],
        }
        report_dict = dict_filter(report_dict, lambda k, v: v != 0.0)
        return report_dict

    def start_training(self):
        """
        main function to call for training
        :return:
        """
        for epoch in range(self._start_epoch, self.max_epoch):
            self._train_loop(
                train_loader_A=self.train_loader_A,
                train_loader_B=self.train_loader_B,
                epoch=epoch,
                head_control_param=self.head_control_params,
            )
            with torch.no_grad():
                current_score = self._eval_loop(self.val_loader, epoch)
            self.METERINTERFACE.step()
            self.model.schedulerStep()
            self.METERINTERFACE.summary().to_csv(self.save_dir / f"wholeMeter.csv")

            self.writer.add_scalars(
                "Scalars",
                self.METERINTERFACE.summary().iloc[-1].to_dict(),
                global_step=epoch,
            )
            self.drawer.draw(self.METERINTERFACE.summary())
            self.save_checkpoint(self.state_dict, epoch, current_score)

    def _train_loop(
        self,
        train_loader_A: DataLoader = None,
        train_loader_B: DataLoader = None,
        epoch: int = 0,
        mode: ModelMode = ModelMode.TRAIN,
        head_control_param={},
        *args,
        **kwargs,
    ):
        """
        :param train_loader: TrainLoader which is the same as the Val
        :param epoch: current epoch
        :param mode: should be ModelMode.TRAIN
        :param args: to ignore
        :param kwargs: to ignore
        :return:
        """
        assert (
            head_control_param.__len__() > 0
        ), f"`head_control_param` must be provided, given {head_control_param}."
        assert set(head_control_param.keys()) <= {"A", "B"}, (
            f"`head_control_param` key must be in `A` or `B`,"
            f" given{set(head_control_param.keys())}"
        )
        for k, v in head_control_param.items():
            assert k in ("A", "B"), (
                f"`head_control_param` key must be in `A` or `B`,"
                f" given{set(head_control_param.keys())}"
            )
            assert isinstance(v, int) and v >= 0, f"Iteration for {k} must be >= 0."
        self.model.set_mode(mode)
        assert (
            self.model.training
        ), f"Model should be in train() model, given {self.model.training}."
        """
        assert len(train_loader_B) == len(train_loader_A), f"The length of the train_loaders should be the same,\"" \
            f"given `len(train_loader_A)`:{len(train_loader_A)} and `len(train_loader_B)`:{len(train_loader_B)}."
        This assert is not always true for STL-10 dataset.
        """
        for head_name, head_iterations in head_control_param.items():
            assert head_name in ("A", "B"), head_name
            train_loader = eval(
                f"train_loader_{head_name}"
            )  # change the dataloader for different head
            for head_epoch in range(head_iterations):
                # given one head, one iteration in this head, and one train_loader.
                train_loader_: tqdm = tqdm_(
                    train_loader
                )  # reinitialize the train_loader
                train_loader_.set_description(
                    f"Training epoch: {epoch} head:{head_name}, head_epoch:{head_epoch + 1}/{head_iterations}"
                )
                # time_before = time.time()
                for batch, image_labels in enumerate(train_loader_):
                    images, *_ = list(zip(*image_labels))
                    tf1_images = torch.cat(
                        tuple([images[0] for _ in range(images.__len__() - 1)]), dim=0
                    ).to(self.device)
                    tf2_images = torch.cat(tuple(images[1:]), dim=0).to(self.device)
                    if self.use_sobel:
                        tf1_images = self.sobel(tf1_images)
                        tf2_images = self.sobel(tf2_images)
                    assert tf1_images.shape == tf2_images.shape
                    batch_loss = self._trainer_specific_loss(
                        tf1_images, tf2_images, head_name
                    )
                    with ZeroGradientBackwardStep(batch_loss, self.model) as loss:
                        loss.backward()
                    report_dict = self._training_report_dict
                    train_loader_.set_postfix(report_dict)

        report_dict_str = ", ".join([f"{k}:{v:.3f}" for k, v in report_dict.items()])
        print(f"Training epoch: {epoch} : {report_dict_str}")

    def _eval_loop(
        self,
        val_loader: DataLoader = None,
        epoch: int = 0,
        mode: ModelMode = ModelMode.EVAL,
        *args,
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
            images, gt = list(zip(*image_labels))
            images, gt = images[0].to(self.device), gt[0].to(self.device)
            if self.use_sobel:
                images = self.sobel(images)
            _pred = self.model.torchnet(images, head="B")
            assert _pred.__len__() == self.model.arch_dict["num_sub_heads"]
            assert simplex(_pred[0]), f"pred should be normalized, given {_pred[0]}."
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
            self.METERINTERFACE.val_average_acc.add(_acc)
        # record best acc
        self.METERINTERFACE.val_best_acc.add(max(subhead_accs))
        self.METERINTERFACE.val_worst_acc.add(min(subhead_accs))
        report_dict = self._eval_report_dict

        report_dict_str = ", ".join([f"{k}:{v:.3f}" for k, v in report_dict.items()])
        print(f"Validating epoch: {epoch} : {report_dict_str}")
        return self.METERINTERFACE.val_best_acc.summary()["mean"]

    def _trainer_specific_loss(self, tf1_images, tf2_images, head_name):
        self.model.zero_grad()
        tf1_pred_simplex = self.model.torchnet(tf1_images, head=head_name)
        tf2_pred_simplex = self.model.torchnet(tf2_images, head=head_name)
        assert (
            simplex(tf1_pred_simplex[0])
            and tf1_pred_simplex.__len__() == tf2_pred_simplex.__len__()
        )
        batch_loss: List[torch.Tensor] = []  # type: ignore
        for subhead in range(tf1_pred_simplex.__len__()):
            _loss, _loss_no_lambda = self.criterion(
                tf1_pred_simplex[subhead], tf2_pred_simplex[subhead]
            )
            batch_loss.append(_loss)
        batch_loss: torch.Tensor = sum(batch_loss) / len(batch_loss)
        self.METERINTERFACE[f"train_head_{head_name}"].add(
            -batch_loss.item()
        )  # type: ignore

        return batch_loss
