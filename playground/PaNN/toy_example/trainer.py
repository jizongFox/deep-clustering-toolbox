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
from .augment import AffineTensorTransform


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
            (unlab_img, unlab_gt) in zip(_max_iter, labeled_loader, unlabeled_loader):
            lab_img, lab_gt = lab_img.to(self.device), lab_gt.to(self.device)
            lab_preds = self.model(lab_img)
            sup_loss = self.kl_criterion(
                lab_preds,
                class2one_hot(
                    lab_gt,
                    C=self.model.torchnet.num_classes
                ).float()
            )
            reg_loss = self._trainer_specific_loss(unlab_img, unlab_gt)
            self.METERINTERFACE["traloss"].add(sup_loss.item())
            self.METERINTERFACE["traconf"].add(lab_preds.max(1)[1], lab_gt)

            with ZeroGradientBackwardStep(sup_loss + reg_loss, self.model) as total_loss:
                total_loss.backward()
            report_dict = self._training_report_dict
            _max_iter.set_postfix(report_dict)
        print(f"Training Epoch {epoch}: {nice_dict(report_dict)}")
        self.writer.add_scalar_with_tag("train", report_dict, global_step=epoch)

    def _trainer_specific_loss(self, unlab_img: Tensor, unlab_gt: Tensor, **kwargs) -> Tensor:
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


class WeightedEntropy(nn.Module):
    r"""General Entropy interface

    the definition of Entropy is - \sum p(xi) log (p(xi))

    reduction (string, optional): Specifies the reduction to apply to the output:
    ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
    ``'mean'``: the sum of the output will be divided by the number of
    elements in the output, ``'sum'``: the output will be summed.
    """

    def __init__(self, reduction="mean", eps=1e-16, weight=None):
        super().__init__()
        from deepclustering.loss.loss import _check_reduction_params
        _check_reduction_params(reduction)
        self._eps = eps
        self._reduction = reduction
        if weight is None:
            self._weight = None
        else:
            if isinstance(weight, Tensor):
                assert torch.allclose(weight.sum(), torch.tensor(1.0, dtype=torch.float32, device=weight.device))
            elif isinstance(weight, list):
                assert sum(weight) == 1
            else:
                raise TypeError("weight should be Tensor or list.")
            self._weight = weight if isinstance(weight, Tensor) else Tensor(weight).float()

    def forward(self, input: Tensor) -> Tensor:
        assert input.shape.__len__() >= 2
        b, c, *s = input.shape
        assert simplex(input), f"Entropy input should be a simplex"

        if self._weight is not None:
            assert self._weight.shape[0] == c
            weight_matrix = torch.zeros_like(input, dtype=input.dtype, device=input.device)
            for i, w in enumerate(self._weight):
                weight_matrix[:, i] = w / input[:, i].detach()
            e = -1.0 * (input * weight_matrix).detach() * (
                        input + self._eps).log()  # cross entropy, which is not 0 for minial
        else:
            e = input * (input + self._eps).log()
        e = -1.0 * e.sum(1)
        assert e.shape == torch.Size([b, *s])
        if self._reduction == "mean":
            return e.mean()
        elif self._reduction == "sum":
            return e.sum()
        else:
            return e


class WeightedIIC:

    def __init__(self, weight=None) -> None:
        super().__init__()
        self.weighted_entropy = WeightedEntropy(weight=weight)
        from deepclustering.loss.IID_losses import compute_joint
        self.compute_joint = compute_joint

    def __call__(self, x_out1: Tensor, x_out2: Tensor):
        assert simplex(x_out1) and simplex(x_out2)
        joint_distr = self.compute_joint(x_out1, x_out2)
        mi = self.weighted_entropy(joint_distr.sum(0).unsqueeze(0)) + (
                    joint_distr * (joint_distr / joint_distr.sum(1, keepdim=True)).log()).sum()
        return mi * -1.0


class SemiWeightedIICTrainer(SemiTrainer):

    @lazy_load_checkpoint
    def __init__(self, model: Model, labeled_loader: DataLoader, unlabeled_loader: DataLoader, val_loader: DataLoader,
                 max_epoch: int = 100, save_dir: str = "base", checkpoint_path: str = None, device="cpu",
                 config: dict = None, max_iter: int = 100, prior=None, **kwargs) -> None:
        super().__init__(model, labeled_loader, unlabeled_loader, val_loader, max_epoch, save_dir, checkpoint_path,
                         device, config, max_iter, **kwargs)
        self.prior = prior
        self.weighted_iic_criterion = WeightedIIC(weight=prior)
        self.affine_transform = AffineTensorTransform(min_rot=0, max_rot=15, min_scale=.8, max_scale=1.2, )

    def __init_meters__(self) -> List[Union[str, List[str]]]:
        columns = super().__init_meters__()
        self.METERINTERFACE.register_new_meter("marginal", AverageValueMeter())
        self.METERINTERFACE.register_new_meter("mi", AverageValueMeter())
        self.METERINTERFACE.register_new_meter("unl_acc", ConfusionMatrix(5))
        columns.extend(["marginal_mean", "mi_mean"])
        return columns

    def _trainer_specific_loss(self, unlab_img: Tensor, unlab_gt: Tensor, **kwargs) -> Tensor:
        unlab_img = unlab_img.to(self.device)
        unlab_img_tf, _ = self.affine_transform(unlab_img)
        all_preds = self.model(torch.cat([unlab_img, unlab_img_tf], dim=0))
        unlabel_pred, unlabel_pred_tf = torch.chunk(all_preds, 2)
        assert simplex(unlabel_pred) and simplex(unlabel_pred_tf)
        mi = self.weighted_iic_criterion(unlabel_pred, unlabel_pred_tf)
        # mi, _ = IIDLoss(lamb=1)(unlabel_pred, unlabel_pred_tf)
        self.METERINTERFACE["mi"].add(-mi.item())
        self.METERINTERFACE["unl_acc"].add(unlabel_pred.max(1)[1], unlab_gt)
        marginal = unlabel_pred.mean(0)
        if self.prior is not None:
            marginal_loss = self.kl_criterion(marginal.unsqueeze(0), self.prior.unsqueeze(0).to(self.device))
            self.METERINTERFACE["marginal"].add(marginal_loss.item())
        return mi

    @property
    def _training_report_dict(self):
        report_dict = super()._training_report_dict
        report_dict.update({
            "unl_acc": self.METERINTERFACE["unl_acc"].summary()["acc"],
            "mi": self.METERINTERFACE["mi"].summary()["mean"],
            "marginal": self.METERINTERFACE["marginal"].summary()["mean"]
        })
        return filter_dict(report_dict)
