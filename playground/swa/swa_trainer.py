import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Union
from torch.utils.data import DataLoader

from deepclustering import ModelMode
from deepclustering.model import Model, ZeroGradientBackwardStep
from deepclustering.trainer import _Trainer
from deepclustering.meters import MeterInterface, AverageValueMeter, ConfusionMatrix
from deepclustering.decorator import lazy_load_checkpoint
from deepclustering.utils import filter_dict, tqdm_


class SWATrainer(_Trainer):
    @lazy_load_checkpoint
    def __init__(self, model: Model, train_loader: DataLoader, val_loader: DataLoader, max_epoch: int = 100,
                 save_dir: str = "swa_trainer", checkpoint_path: str = None, device="cpu", config: dict = None,
                 **kwargs) -> None:
        super().__init__(model, train_loader, val_loader, max_epoch, save_dir, checkpoint_path, device, config,
                         **kwargs)
        # initialize swa_model.
        self.swa_model = Model.initialize_from_state_dict(model.state_dict())
        # criterion
        self.ce_criterion = nn.CrossEntropyLoss()

    def __init_meters__(self) -> List[Union[str, List[str]]]:
        METER_CONFIG = {"train_loss": AverageValueMeter(),
                        "val_loss": AverageValueMeter(),
                        "val_swa_loss": AverageValueMeter(),
                        "train_acc": ConfusionMatrix(10),
                        "val_acc": ConfusionMatrix(10),
                        "val_swa_acc": ConfusionMatrix(10)
                        }
        self.METERINTERFACE = MeterInterface(METER_CONFIG)  # type:ignore
        return [["train_loss", "val_loss", "val_swa_loss"], ["train_acc", "val_acc", "val_swa_acc"]]

    @property
    def _training_report_dict(self):
        return filter_dict({"tra_loss": self.METERINTERFACE["train_loss"].summary()["mean"],
                            "tra_acc": self.METERINTERFACE["train_acc"].summary()["acc"]})

    @property
    def _eval_report_dict(self):
        return filter_dict({"val_loss": self.METERINTERFACE["val_loss"].summary()["mean"],
                            "val_acc": self.METERINTERFACE["val_acc"].summary()["acc"]})

    def _train_loop(
            self,
            train_loader: DataLoader = None,
            epoch: int = 0,
            mode=ModelMode.TRAIN,
            *args,
            **kwargs
    ):
        # set model mode
        self.model.set_mode(mode)
        assert self.model.torchnet.training == True
        # set tqdm-based trainer
        _train_loader = tqdm_(train_loader)
        _train_loader.set_description(f" Training epoch {epoch}: ")
        for batch_id, (imgs, targets) in enumerate(_train_loader):
            imgs, targets = imgs.to(self.device), targets.to(self.device)
            preds = self.model(imgs)
            loss = self.ce_criterion(preds, targets)
            with ZeroGradientBackwardStep(loss, self.model) as scaled_loss:
                scaled_loss.backward()

            self.METERINTERFACE["train_loss"].add(loss.item())
            self.METERINTERFACE["train_acc"].add(preds.max(1)[1], targets)
            report_dict = self._training_report_dict
            _train_loader.set_postfix(report_dict)

    def _eval_loop(
            self,
            val_loader: DataLoader = None,
            epoch: int = 0,
            mode=ModelMode.EVAL,
            *args,
            **kwargs
    ) -> float:
        # set model mode
        self.model.set_mode(mode)
        assert self.model.torchnet.training == False, self.model.training
        # set tqdm-based trainer
        _val_loader = tqdm_(val_loader)
        _val_loader.set_description(f"Validating epoch {epoch}: ")
        for batch_id, (imgs, targets) in enumerate(_val_loader):
            imgs, targets = imgs.to(self.device), targets.to(self.device)
            preds = self.model(imgs)
            loss = self.ce_criterion(preds, targets)
            self.METERINTERFACE["val_loss"].add(loss.item())
            self.METERINTERFACE["val_acc"].add(preds.max(1)[1], targets)
            report_dict = self._eval_report_dict
            _val_loader.set_postfix(report_dict)
        return self.METERINTERFACE["val_acc"].summary()["acc"]

    def _eval_swa_loop(
            self,
            val_loader: DataLoader = None,
            epoch: int = 0,
            mode=ModelMode.EVAL,
    ) -> float:
        # set model mode
        self.swa_model.set_mode(mode)
        assert self.swa_model.torchnet.training == False, self.swa_model.training
        # set tqdm-based trainer
        _val_loader = tqdm_(val_loader)
        _val_loader.set_description(f"Validating SWA epoch {epoch}: ")
        for batch_id, (imgs, targets) in enumerate(_val_loader):
            imgs, targets = imgs.to(self.device), targets.to(self.device)
            preds = self.swa_model(imgs)
            loss = self.ce_criterion(preds, targets)
            self.METERINTERFACE["val_swa_loss"].add(loss.item())
            self.METERINTERFACE["val_swa_acc"].add(preds.max(1)[1], targets)
            report_dict = self._eval_report_dict
            _val_loader.set_postfix(report_dict)
        return self.METERINTERFACE["val_swa_acc"].summary()["acc"]

    def step_swa(self):
        self._moving_average(self.swa_model, self.model)

    @staticmethod
    def _moving_average(net1: Model, net2: Model, alpha=1):
        for param1, param2 in zip(net1.torchnet.parameters(), net2.torchnet.parameters()):
            param1.data *= (1.0 - alpha)
            param1.data += param2.data * alpha
