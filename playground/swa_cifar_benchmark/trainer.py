from typing import List, Union

from termcolor import colored
from torch import nn
from torch.utils.data import DataLoader

from deepclustering import ModelMode
from deepclustering.decorator import lazy_load_checkpoint
from deepclustering.meters import MeterInterface, AverageValueMeter, ConfusionMatrix, InstanceValue
from deepclustering.model import Model, ZeroGradientBackwardStep
from deepclustering.trainer import _Trainer
from deepclustering.utils import filter_dict, tqdm_, nice_dict


class SGDTrainer(_Trainer):
    @lazy_load_checkpoint
    def __init__(self, model: Model, train_loader: DataLoader, val_loader: DataLoader, max_epoch: int = 100,
                 save_dir: str = "swa_trainer", checkpoint_path: str = None, device="cpu", config: dict = None,
                 **kwargs) -> None:
        super().__init__(model, train_loader, val_loader, max_epoch, save_dir, checkpoint_path, device, config,
                         **kwargs)
        self.ce_criterion = nn.CrossEntropyLoss()

    def __init_meters__(self) -> List[Union[str, List[str]]]:
        METER_CONFIG = {"lr": InstanceValue(),
                        "train_loss": AverageValueMeter(),
                        "val_loss": AverageValueMeter(),
                        "train_acc": ConfusionMatrix(10),
                        "val_acc": ConfusionMatrix(10),
                        }
        self.METERINTERFACE = MeterInterface(METER_CONFIG)  # type:ignore
        return [["train_loss_mean", "val_loss_mean"],
                ["train_acc_acc", "val_acc_acc"], "lr_value"]

    @property
    def _training_report_dict(self):
        return filter_dict({"tra_loss": self.METERINTERFACE["train_loss"].summary()["mean"],
                            "tra_acc": self.METERINTERFACE["train_acc"].summary()["acc"]})

    @property
    def _eval_report_dict(self):
        return filter_dict({"val_loss": self.METERINTERFACE["val_loss"].summary()["mean"],
                            "val_acc": self.METERINTERFACE["val_acc"].summary()["acc"]})

    def _train_loop(self, train_loader: DataLoader = None, epoch: int = 0, mode=ModelMode.TRAIN, *args, **kwargs):
        # set model mode
        self.model.set_mode(mode)
        assert self.model.torchnet.training == True
        # set tqdm-based trainer
        self.METERINTERFACE["lr"].add(self.model.get_lr()[0])
        _train_loader = tqdm_(train_loader)
        _train_loader.set_description(f" Training epoch {epoch}: lr={self.METERINTERFACE['lr'].summary()['value']:.5f}")
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
        print(colored(f"  Training epoch {epoch}: {nice_dict(report_dict)}", "red"))

    def _eval_loop(self, val_loader: DataLoader = None, epoch: int = 0, mode=ModelMode.EVAL, *args, **kwargs) -> float:
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
        print(colored(f"Validating epoch {epoch}: {nice_dict(report_dict)}", "green"))
        return self.METERINTERFACE["val_acc"].summary()["acc"]


class SWATrainer(SGDTrainer):

    def __init__(self, model: Model, train_loader: DataLoader, val_loader: DataLoader, max_epoch: int = 100,
                 save_dir: str = "swa_trainer", checkpoint_path: str = None, device="cpu", config: dict = None,
                 **kwargs) -> None:
        super().__init__(model, train_loader, val_loader, max_epoch, save_dir, checkpoint_path, device, config,
                         **kwargs)
        # initialize swa_model.
        self.swa_model = Model.initialize_from_state_dict(model.state_dict())
        self.swa_model.to(self.device)

    def __init_meters__(self) -> List[Union[str, List[str]]]:
        _ = super().__init_meters__()
        self.METERINTERFACE.register_new_meter("train_swa_loss", AverageValueMeter())
        self.METERINTERFACE.register_new_meter("train_swa_acc", AverageValueMeter())
        self.METERINTERFACE.register_new_meter("val_swa_loss", AverageValueMeter())
        self.METERINTERFACE.register_new_meter("val_swa_acc", ConfusionMatrix(10))
        self.METERINTERFACE.register_new_meter("train_swa_acc", ConfusionMatrix(10))
        return [["train_loss_mean", "val_loss_mean"], ["train_swa_loss_mean", "val_swa_loss_mean"],
                ["train_acc_acc", "val_acc_acc"], ["train_swa_acc_acc", "val_swa_acc_acc"], "lr_value"]

    def start_training(self):
        import torch
        for epoch in range(self._start_epoch, self.max_epoch):
            self._train_loop(train_loader=self.train_loader, epoch=epoch)
            with torch.no_grad():
                current_score = self._eval_loop(self.val_loader, epoch)
                # inference on swa model
                if True:  # if some condition
                    self.step_swa()
                    self._eval_swa_loop(self.val_loader, epoch)
                # inference on swa ends

            self.METERINTERFACE.step()
            self.model.schedulerStep()

            # save meters and checkpoints
            SUMMARY = self.METERINTERFACE.summary()
            SUMMARY.to_csv(self.save_dir / self.wholemeter_filename)
            self.drawer.draw(SUMMARY)
            self.save_checkpoint(self.state_dict(), epoch, current_score)

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
        print(colored(f"Validating epoch {epoch}: SWA -> {nice_dict(report_dict)}", "blue"))
        return self.METERINTERFACE["val_swa_acc"].summary()["acc"]

    def step_swa(self):
        self._moving_average(self.swa_model, self.model)
        self._adjust_bn()

    @staticmethod
    def _adjust_bn():
        pass

    @staticmethod
    def _moving_average(net1: Model, net2: Model, alpha=1):
        for param1, param2 in zip(net1.torchnet.parameters(), net2.torchnet.parameters()):
            param1.data *= (1.0 - alpha)
            param1.data += param2.data * alpha
