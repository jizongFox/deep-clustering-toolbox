from typing import *

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from deepclustering import ModelMode
from deepclustering.decorator import lazy_load_checkpoint
from deepclustering.loss import KL_div, simplex, Entropy
from deepclustering.meters import AverageValueMeter, MeterInterface, ConfusionMatrix
from deepclustering.model import Model, ZeroGradientBackwardStep
from deepclustering.optim import RAdam
from deepclustering.trainer import _Trainer
from deepclustering.utils import class2one_hot, tqdm_, flatten_dict, nice_dict, filter_dict


class SemiTrainer(_Trainer):

    @lazy_load_checkpoint
    def __init__(self, model: Model, labeled_loader: DataLoader, unlabeled_loader: DataLoader, val_loader: DataLoader,
                 max_epoch: int = 100, save_dir: str = "base", checkpoint_path: str = None, device="cpu",
                 config: dict = None, max_iter: int = 100, **kwargs) -> None:
        super().__init__(model, None, val_loader, max_epoch, save_dir, checkpoint_path, device, config, **kwargs)
        assert self.train_loader == None
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.kl_criterion = KL_div()
        self.max_iter = max_iter

    def __init_meters__(self) -> List[Union[str, List[str]]]:
        meter_config = {
            "lr": AverageValueMeter(),
            "traloss": AverageValueMeter(),
            "traconf": ConfusionMatrix(self.model.torchnet.num_classes),
            "valloss": AverageValueMeter(),
            "valconf": ConfusionMatrix(self.model.torchnet.num_classes),
        }
        self.METERINTERFACE = MeterInterface(meter_config)
        return ["traloss_mean", "traconf_acc", "valloss_mean", "valconf_acc", "lr_mean"]

    def start_training(self):
        for epoch in range(self._start_epoch, self.max_epoch):
            self._train_loop(labeled_loader=self.labeled_loader, unlabeled_loader=self.unlabeled_loader, epoch=epoch)
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

    def _train_loop(self, labeled_loader: DataLoader = None, unlabeled_loader: DataLoader = None, epoch: int = 0,
                    mode=ModelMode.TRAIN, *args, **kwargs):
        self.model.set_mode(mode)
        _max_iter = tqdm_(range(self.max_iter))
        _max_iter.set_description(f"Training Epoch {epoch}")
        self.METERINTERFACE["lr"].add(self.model.get_lr()[0])
        for batch_num, (lab_img, lab_gt), \
            (unlab_img, _) in zip(_max_iter, labeled_loader, unlabeled_loader):
            lab_img, lab_gt = lab_img.to(self.device), lab_gt.to(self.device)
            lab_preds = self.model(lab_img)
            sup_loss = self.kl_criterion(
                lab_preds,
                class2one_hot(
                    lab_gt,
                    C=self.model.torchnet.num_classes
                ).float()
            )
            reg_loss = self._trainer_specific_loss(unlab_img)
            self.METERINTERFACE["traloss"].add(sup_loss.item())
            self.METERINTERFACE["traconf"].add(lab_preds.max(1)[1], lab_gt)

            with ZeroGradientBackwardStep(sup_loss + reg_loss, self.model) as total_loss:
                total_loss.backward()
            report_dict = self._training_report_dict
            _max_iter.set_postfix(report_dict)
        print(f"Training Epoch {epoch}: {nice_dict(report_dict)}")
        self.writer.add_scalar_with_tag("train", report_dict, global_step=epoch)

    def _trainer_specific_loss(self, unlab_img: Tensor, **kwargs) -> Tensor:
        return torch.tensor(0, dtype=torch.float32, device=self.device)

    def _eval_loop(self, val_loader: DataLoader = None, epoch: int = 0, mode=ModelMode.EVAL, *args, **kwargs) -> float:
        self.model.set_mode(mode)
        _val_loader = tqdm_(val_loader)
        _val_loader.set_description(f"Validating Epoch {epoch}")
        for batch_num, (val_img, val_gt) in enumerate(_val_loader):
            val_img, val_gt = val_img.to(self.device), val_gt.to(self.device)
            val_preds = self.model(val_img)
            val_loss = self.kl_criterion(
                val_preds,
                class2one_hot(
                    val_gt,
                    C=self.model.torchnet.num_classes
                ).float(),
                disable_assert=True
            )
            self.METERINTERFACE["valloss"].add(val_loss.item())
            self.METERINTERFACE["valconf"].add(val_preds.max(1)[1], val_gt)
            report_dict = self._eval_report_dict
            _val_loader.set_postfix(report_dict)
        print(f"Validating Epoch {epoch}: {nice_dict(report_dict)}")
        self.writer.add_scalar_with_tag(tag="eval", tag_scalar_dict=report_dict, global_step=epoch)
        return self.METERINTERFACE["valconf"].summary()["acc"]

    @property
    def _training_report_dict(self):
        return flatten_dict({"tra_loss": self.METERINTERFACE["traloss"].summary()["mean"],
                             "tra_acc": self.METERINTERFACE["traconf"].summary()["acc"],
                             "lr": self.METERINTERFACE["lr"].summary()["mean"]}, sep="_")

    @property
    def _eval_report_dict(self):
        return flatten_dict(
            {"val_loss": self.METERINTERFACE["valloss"].summary()["mean"],
             "val_acc": self.METERINTERFACE["valconf"].summary()['acc']
             }, sep="")


class SemiEntropyTrainer(SemiTrainer):

    @lazy_load_checkpoint
    def __init__(self, model: Model, labeled_loader: DataLoader, unlabeled_loader: DataLoader, val_loader: DataLoader,
                 max_epoch: int = 100, save_dir: str = "base", checkpoint_path: str = None, device="cpu",
                 config: dict = None, max_iter: int = 100, prior: Tensor = None, use_centropy=False, **kwargs) -> None:
        super().__init__(model, labeled_loader, unlabeled_loader, val_loader, max_epoch, save_dir, checkpoint_path,
                         device, config, max_iter, **kwargs)
        self.prior = prior.to(self.device)
        self.use_centropy = use_centropy
        self.entropy = Entropy()

    def __init_meters__(self) -> List[Union[str, List[str]]]:
        columns = super().__init_meters__()
        self.METERINTERFACE.register_new_meter("marginal", AverageValueMeter())
        self.METERINTERFACE.register_new_meter("centropy", AverageValueMeter())
        columns.extend(["marginal_mean", "centropy_mean"])
        return columns

    def _trainer_specific_loss(self, unlab_img: Tensor, **kwargs) -> Tensor:
        unlab_img = unlab_img.to(self.device)
        unlabeled_preds = self.model(unlab_img)
        assert simplex(unlabeled_preds, 1)
        marginal = unlabeled_preds.mean(0)
        marginal_loss = self.kl_criterion(marginal.unsqueeze(0), self.prior.unsqueeze(0))
        self.METERINTERFACE["marginal"].add(marginal_loss.item())
        if self.use_centropy:
            centropy = self.entropy(unlabeled_preds)
            marginal_loss += centropy
            self.METERINTERFACE["centropy"].add(centropy.item())

        return marginal_loss

    @property
    def _training_report_dict(self):
        report_dict = super()._training_report_dict
        report_dict.update({
            "marginal": self.METERINTERFACE["marginal"].summary()["mean"],
            "centropy": self.METERINTERFACE["centropy"].summary()["mean"]
        })
        return filter_dict(report_dict)


class SemiPrimalDualTrainer(SemiEntropyTrainer):
    @lazy_load_checkpoint
    def __init__(self, model: Model, labeled_loader: DataLoader, unlabeled_loader: DataLoader, val_loader: DataLoader,
                 max_epoch: int = 100, save_dir: str = "base", checkpoint_path: str = None, device="cpu",
                 config: dict = None, max_iter: int = 100, prior: Tensor = None, use_centropy=False, **kwargs) -> None:
        super().__init__(model, labeled_loader, unlabeled_loader, val_loader, max_epoch, save_dir, checkpoint_path,
                         device, config, max_iter, prior, use_centropy, **kwargs)
        self.mu = nn.Parameter(-1.0 / self.prior)  # initialize mu = - 1 / prior
        self.mu_optim = RAdam((self.mu,), lr=1e-4, betas=(0.5, 0.999))

    def __init_meters__(self) -> List[Union[str, List[str]]]:
        columns = super().__init_meters__()
        self.METERINTERFACE.register_new_meter("residual", AverageValueMeter())
        columns.append("residual_mean")
        return columns

    def _trainer_specific_loss(self, unlab_img: Tensor, **kwargs) -> Tensor:
        unlab_img = unlab_img.to(self.device)
        unlabeled_preds = self.model(unlab_img)
        assert simplex(unlabeled_preds, 1)
        marginal = unlabeled_preds.mean(0)
        lagrangian = (self.prior * (marginal * self.mu.detach() + 1 + (-self.mu.detach()).log())).sum()
        if self.use_centropy:
            centropy = self.entropy(unlabeled_preds)
            self.METERINTERFACE["centropy"].add(centropy.item())
            return lagrangian + centropy
        return lagrangian

    def _update_mu(self, unlab_img: Tensor):
        self.mu_optim.zero_grad()
        unlab_img = unlab_img.to(self.device)
        unlabeled_preds = self.model(unlab_img).detach()
        assert simplex(unlabeled_preds, 1)
        marginal = unlabeled_preds.mean(0)
        lagrangian = (self.prior * (marginal * self.mu + 1 + (-self.mu).log())).sum()
        lagrangian.backward()
        self.mu_optim.step()

        with torch.no_grad():
            self.mu += self.mu.grad
            self.METERINTERFACE["residual"].add(self.mu.grad.abs().sum().item())
            # to quantify:
            marginal_loss = self.kl_criterion(marginal.unsqueeze(0), self.prior.unsqueeze(0), disable_assert=True)
            self.METERINTERFACE["marginal"].add(marginal_loss.item())

    def _train_loop(self, labeled_loader: DataLoader = None, unlabeled_loader: DataLoader = None, epoch: int = 0,
                    mode=ModelMode.TRAIN, *args, **kwargs):
        self.model.set_mode(mode)
        _max_iter = tqdm_(range(self.max_iter))
        _max_iter.set_description(f"Training Epoch {epoch}")
        self.METERINTERFACE["lr"].add(self.model.get_lr()[0])
        for batch_num, (lab_img, lab_gt), \
            (unlab_img, _) in zip(_max_iter, labeled_loader, unlabeled_loader):
            lab_img, lab_gt = lab_img.to(self.device), lab_gt.to(self.device)
            lab_preds = self.model(lab_img)
            sup_loss = self.kl_criterion(
                lab_preds,
                class2one_hot(
                    lab_gt,
                    C=self.model.torchnet.num_classes
                ).float()
            )
            reg_loss = self._trainer_specific_loss(unlab_img)
            self.METERINTERFACE["traloss"].add(sup_loss.item())
            self.METERINTERFACE["traconf"].add(lab_preds.max(1)[1], lab_gt)

            with ZeroGradientBackwardStep(sup_loss + reg_loss, self.model) as total_loss:
                total_loss.backward()

            self._update_mu(unlab_img)
            report_dict = self._training_report_dict
            _max_iter.set_postfix(report_dict)
        print(f"Training Epoch {epoch}: {nice_dict(report_dict)}")
        self.writer.add_scalar_with_tag("train", report_dict, global_step=epoch)

    @property
    def _training_report_dict(self):
        report_dict = super()._training_report_dict
        report_dict.update(
            {"residual": self.METERINTERFACE["residual"].summary()["mean"]}
        )
        return report_dict
