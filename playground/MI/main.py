###############################
#   This file is to create experiments using Mnist dataset of IIC setting (single head)
#   to verify whether the IMSAT (with VAT or random perturbation) or the IIC (baseline or vat) perform best in the
#   simplest case.
#
##############################
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from deepclustering import ModelMode
from deepclustering.dataset.classification.mnist_helper import (
    MNISTClusteringDatasetInterface,
    default_mnist_img_transform,
)
from deepclustering.loss.IMSAT_loss import MultualInformaton_IMSAT
from deepclustering.loss.loss import JSD_div, KL_div
from deepclustering.loss.IID_losses import IIDLoss
from deepclustering.manager import ConfigManger
from deepclustering.meters import MeterInterface, AverageValueMeter
from deepclustering.model import Model
from deepclustering.trainer.Trainer import _Trainer
from deepclustering.utils import tqdm_, simplex, tqdm, flatten_dict, dict_filter
from deepclustering.utils.VAT import VATLoss_Multihead
from deepclustering.utils.classification.assignment_mapping import (
    hungarian_match,
    flat_acc,
)

matplotlib.use("agg")


class IMSAT_Trainer(_Trainer):
    """
    MI(x,p) + CE(p,adv(p)) or MI(x,p) + CE(p,geom(p))
    """

    def __init__(
        self,
        model: Model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epoch: int = 100,
        save_dir: str = "IMSAT",
        use_vat: bool = False,
        sat_weight: float = 0.1,
        checkpoint_path: str = None,
        device="cpu",
        config: dict = None,
        **kwargs,
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
            **kwargs,
        )
        self.use_vat = use_vat
        self.sat_weight = float(sat_weight)
        self.criterion = MultualInformaton_IMSAT(mu=4, separate_return=True)
        self.jsd = JSD_div()
        self.kl = KL_div()
        plt.ion()

    def __init_meters__(self) -> List[str]:
        METER_CONFIG = {
            "train_mi": AverageValueMeter(),
            "train_entropy": AverageValueMeter(),
            "train_centropy": AverageValueMeter(),
            "train_sat": AverageValueMeter(),
            "val_best_acc": AverageValueMeter(),
            "val_average_acc": AverageValueMeter(),
        }
        self.METERINTERFACE = MeterInterface(METER_CONFIG)
        return [
            "train_mi_mean",
            "train_entropy_mean",
            "train_centropy_mean",
            "train_sat_mean",
            "val_average_acc_mean",
            "val_best_acc_mean",
        ]

    @property
    def _training_report_dict(self):
        report_dict = dict_filter(
            flatten_dict(
                {
                    "train_MI": self.METERINTERFACE.train_mi.summary()["mean"],
                    "train_entropy": self.METERINTERFACE.train_entropy.summary()[
                        "mean"
                    ],
                    "train_centropy": self.METERINTERFACE.train_centropy.summary()[
                        "mean"
                    ],
                    "train_sat": self.METERINTERFACE.train_sat.summary()["mean"],
                }
            ),
            lambda k, v: v != 0.0,
        )
        return report_dict

    @property
    def _eval_report_dict(self):
        report_dict = flatten_dict(
            {
                "average_acc": self.METERINTERFACE.val_average_acc.summary()["mean"],
                "best_acc": self.METERINTERFACE.val_best_acc.summary()["mean"],
            }
        )
        return report_dict

    def start_training(self):
        for epoch in range(self._start_epoch, self.max_epoch):
            self._train_loop(train_loader=self.train_loader, epoch=epoch)
            with torch.no_grad():
                current_score = self._eval_loop(self.val_loader, epoch)
            self.METERINTERFACE.step()
            self.model.schedulerStep()
            # save meters and checkpoints
            for k, v in self.METERINTERFACE.aggregated_meter_dict.items():
                v.summary().to_csv(self.save_dir / f"meters/{k}.csv")
            self.METERINTERFACE.summary().to_csv(
                self.save_dir / self.wholemeter_filename
            )
            self.writer.add_scalars(
                "Scalars",
                self.METERINTERFACE.summary().iloc[-1].to_dict(),
                global_step=epoch,
            )
            self.drawer.call_draw()
            self.save_checkpoint(self.state_dict, epoch, current_score)

    def _train_loop(
        self,
        train_loader: DataLoader,
        epoch: int,
        mode=ModelMode.TRAIN,
        *args,
        **kwargs,
    ):
        super()._train_loop(*args, **kwargs)
        self.model.set_mode(mode)
        assert self.model.training
        _train_loader: tqdm = tqdm_(train_loader)
        for _batch_num, images_labels_indices in enumerate(_train_loader):
            images, labels, *_ = zip(*images_labels_indices)
            tf1_images = torch.cat(
                tuple([images[0] for _ in range(images.__len__() - 1)]), dim=0
            ).to(self.device)
            tf2_images = torch.cat(tuple(images[1:]), dim=0).to(self.device)
            pred_tf1_simplex = self.model(tf1_images)
            pred_tf2_simplex = self.model(tf2_images)
            assert simplex(pred_tf1_simplex[0]), pred_tf1_simplex
            assert simplex(pred_tf2_simplex[0]), pred_tf2_simplex
            total_loss = self._trainer_specific_loss(
                tf1_images, tf2_images, pred_tf1_simplex, pred_tf2_simplex
            )
            self.model.zero_grad()
            total_loss.backward()
            self.model.step()
            report_dict = self._training_report_dict
            _train_loader.set_postfix(report_dict)

        report_dict_str = ", ".join([f"{k}:{v:.3f}" for k, v in report_dict.items()])
        print(f"  Training epoch: {epoch} : {report_dict_str}")

    def _eval_loop(
        self, val_loader: DataLoader, epoch: int, mode=ModelMode.EVAL, *args, **kwargs
    ) -> float:
        super(IMSAT_Trainer, self)._eval_loop(*args, **kwargs)
        self.model.set_mode(mode)
        assert not self.model.training
        _val_loader = tqdm_(val_loader)
        preds = torch.zeros(
            self.model.arch_dict["num_sub_heads"],
            val_loader.dataset.__len__(),
            dtype=torch.long,
            device=self.device,
        )
        probas = torch.zeros(
            self.model.arch_dict["num_sub_heads"],
            val_loader.dataset.__len__(),
            self.model.arch_dict["output_k"],
            dtype=torch.float,
            device=self.device,
        )
        gts = torch.zeros(
            val_loader.dataset.__len__(), dtype=torch.long, device=self.device
        )
        _batch_done = 0
        for _batch_num, images_labels_indices in enumerate(_val_loader):
            images, labels, *_ = zip(*images_labels_indices)
            images, labels = images[0].to(self.device), labels[0].to(self.device)
            pred = self.model(images)
            _bSlice = slice(_batch_done, _batch_done + images.shape[0])
            gts[_bSlice] = labels
            for subhead in range(pred.__len__()):
                preds[subhead][_bSlice] = pred[subhead].max(1)[1]
                probas[subhead][_bSlice] = pred[subhead]
            _batch_done += images.shape[0]
        assert _batch_done == val_loader.dataset.__len__(), _batch_done

        # record
        subhead_accs = []
        for subhead in range(self.model.arch_dict["num_sub_heads"]):
            reorder_pred, remap = hungarian_match(
                flat_preds=preds[subhead],
                flat_targets=gts,
                preds_k=self.model.arch_dict["output_k"],
                targets_k=self.model.arch_dict["output_k"],
            )
            _acc = flat_acc(reorder_pred, gts)
            subhead_accs.append(_acc)
            # record average acc
            self.METERINTERFACE.val_average_acc.add(_acc)
        self.METERINTERFACE.val_best_acc.add(max(subhead_accs))
        report_dict = self._eval_report_dict

        report_dict_str = ", ".join([f"{k}:{v:.3f}" for k, v in report_dict.items()])
        print(f"Validating epoch: {epoch} : {report_dict_str}")
        return self.METERINTERFACE.val_best_acc.summary()["mean"]

    def _trainer_specific_loss(
        self,
        images: torch.Tensor,
        images_tf: torch.Tensor,
        pred: List[torch.Tensor],
        pred_tf: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        to override
        :param pred:
        :param pred_tf:
        :return:
        """
        assert simplex(pred[0]), pred
        mi_losses, entropy_losses, centropy_losses = [], [], []
        for subhead_num in range(self.model.arch_dict["num_sub_heads"]):
            _mi_loss, (_entropy_loss, _centropy_loss) = self.criterion(
                pred[subhead_num]
            )
            mi_losses.append(-_mi_loss)
            entropy_losses.append(_entropy_loss)
            centropy_losses.append(_centropy_loss)
        mi_loss = sum(mi_losses) / len(mi_losses)
        entrop_loss = sum(entropy_losses) / len(entropy_losses)
        centropy_loss = sum(centropy_losses) / len(centropy_losses)

        self.METERINTERFACE["train_mi"].add(-mi_loss.item())
        self.METERINTERFACE["train_entropy"].add(entrop_loss.item())
        self.METERINTERFACE["train_centropy"].add(centropy_loss.item())

        sat_loss = torch.Tensor([0]).to(self.device)
        if self.sat_weight > 0:
            if not self.use_vat:
                # use transformation
                _sat_loss = list(
                    map(lambda p1, p2: self.kl(p2, p1.detach()), pred, pred_tf)
                )
                sat_loss = sum(_sat_loss) / len(_sat_loss)
            else:
                sat_loss, *_ = VATLoss_Multihead(xi=1, eps=10, prop_eps=0.1)(
                    self.model.torchnet, images
                )

        self.METERINTERFACE["train_sat"].add(sat_loss.item())

        total_loss = mi_loss + self.sat_weight * sat_loss
        return total_loss


class IMSAT_Enhanced_Trainer(IMSAT_Trainer):
    """
    This class is to enhance the IMSAT by enhancing the MI on the original image and transformed image. That means a
    sort of data augmentation.
    MI(x,p) + CE(p,adv(p)) + CE(p,geom(p))
    """

    def __init_meters__(self) -> List[str]:
        METER_CONFIG = {
            "train_mi": AverageValueMeter(),
            "train_entropy": AverageValueMeter(),
            "train_centropy": AverageValueMeter(),
            "train_vat": AverageValueMeter(),
            "train_rt": AverageValueMeter(),
            "val_best_acc": AverageValueMeter(),
            "val_average_acc": AverageValueMeter(),
        }
        self.METERINTERFACE = MeterInterface(METER_CONFIG)
        return [
            "train_mi_mean",
            "train_entropy_mean",
            "train_centropy_mean",
            "train_vat_mean",
            "train_rt_mean",
            "val_average_acc_mean",
            "val_best_acc_mean",
        ]

    @property
    def _training_report_dict(self):
        report_dict = dict_filter(
            flatten_dict(
                {
                    "train_MI": self.METERINTERFACE.train_mi.summary()["mean"],
                    "train_entropy": self.METERINTERFACE.train_entropy.summary()[
                        "mean"
                    ],
                    "train_centropy": self.METERINTERFACE.train_centropy.summary()[
                        "mean"
                    ],
                    "train_vat": self.METERINTERFACE.train_vat.summary()["mean"],
                    "train_rt": self.METERINTERFACE.train_rt.summary()["mean"],
                }
            ),
            lambda k, v: v != 0.0,
        )
        return report_dict

    def _trainer_specific_loss(
        self,
        images: torch.Tensor,
        images_tf: torch.Tensor,
        pred: List[torch.Tensor],
        pred_tf: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        to override
        :param pred:
        :param pred_tf:
        :return:
        """
        assert simplex(pred[0]), pred
        mi_losses, entropy_losses, centropy_losses = [], [], []
        for subhead_num in range(self.model.arch_dict["num_sub_heads"]):
            _mi_loss, (_entropy_loss, _centropy_loss) = self.criterion(
                pred[subhead_num]
            )
            mi_losses.append(-_mi_loss)
            entropy_losses.append(_entropy_loss)
            centropy_losses.append(_centropy_loss)
        mi_loss = sum(mi_losses) / len(mi_losses)
        entrop_loss = sum(entropy_losses) / len(entropy_losses)
        centropy_loss = sum(centropy_losses) / len(centropy_losses)

        self.METERINTERFACE["train_mi"].add(-mi_loss.item())
        self.METERINTERFACE["train_entropy"].add(entrop_loss.item())
        self.METERINTERFACE["train_centropy"].add(centropy_loss.item())

        sat_loss = torch.Tensor([0]).to(self.device)
        if self.sat_weight > 0:
            rt_loss = list(map(lambda p1, p2: self.kl(p2, p1.detach()), pred, pred_tf))
            rt_loss = sum(rt_loss) / len(rt_loss)
            self.METERINTERFACE["train_rt"].add(rt_loss.item())
            vat_loss, *_ = VATLoss_Multihead(xi=1, eps=10, prop_eps=0.1)(
                self.model.torchnet, images
            )
            regul_loss = rt_loss + vat_loss

        total_loss = mi_loss + self.sat_weight * regul_loss
        return total_loss


class IIC_Trainer(IMSAT_Trainer):
    """
    This trainer is to let the VAT as a separated loss to be added to MI.

    MI(p,geom(p)) or  MI(p,geom(p))+kl(p, adv(p))
    """

    def __init__(
        self,
        model: Model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epoch: int = 100,
        save_dir: str = "IMSAT",
        use_vat: bool = False,
        sat_weight: float = 0.1,
        checkpoint_path: str = None,
        device="cpu",
        config: dict = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model,
            train_loader,
            val_loader,
            max_epoch,
            save_dir,
            use_vat,
            sat_weight,
            checkpoint_path,
            device,
            config,
            **kwargs,
        )
        self.criterion = IIDLoss()

    def __init_meters__(self) -> List[str]:
        METER_CONFIG = {
            "train_mi": AverageValueMeter(),
            "train_sat": AverageValueMeter(),
            "val_best_acc": AverageValueMeter(),
            "val_average_acc": AverageValueMeter(),
        }
        self.METERINTERFACE = MeterInterface(METER_CONFIG)
        return [
            "train_mi_mean",
            "train_sat_mean",
            "val_average_acc_mean",
            "val_best_acc_mean",
        ]

    @property
    def _training_report_dict(self):
        report_dict = dict_filter(
            flatten_dict(
                {
                    "train_MI": self.METERINTERFACE.train_mi.summary()["mean"],
                    "train_sat": self.METERINTERFACE.train_sat.summary()["mean"],
                }
            ),
            lambda k, v: v != 0.0,
        )
        return report_dict

    @property
    def _eval_report_dict(self):
        report_dict = dict_filter(
            flatten_dict(
                {
                    "val_average_acc": self.METERINTERFACE.val_average_acc.summary()[
                        "mean"
                    ],
                    "val_best_acc": self.METERINTERFACE.val_best_acc.summary()["mean"],
                }
            ),
            lambda k, v: v != 0.0,
        )
        return report_dict

    def _trainer_specific_loss(
        self,
        images: torch.Tensor,
        images_tf: torch.Tensor,
        pred: List[torch.Tensor],
        pred_tf: List[torch.Tensor],
    ) -> torch.Tensor:
        assert simplex(pred[0]) and pred_tf.__len__() == pred.__len__()
        batch_loss: List[torch.Tensor] = []  # type: ignore
        for subhead in range(pred.__len__()):
            _loss, _loss_no_lambda = self.criterion(pred[subhead], pred_tf[subhead])
            batch_loss.append(_loss)
        batch_loss: torch.Tensor = sum(batch_loss) / len(batch_loss)  # type: ignore
        self.METERINTERFACE[f"train_mi"].add(-batch_loss.item())

        # # vat loss:
        sat_loss = 0
        if self.sat_weight > 0:
            sat_loss, *_ = VATLoss_Multihead(xi=1, eps=10, prop_eps=0.1)(
                self.model.torchnet, images
            )
            self.METERINTERFACE["train_sat"].add(sat_loss.item())
        total_loss = batch_loss + self.sat_weight * sat_loss

        return total_loss


class IIC_enhanced_Trainer(IIC_Trainer):
    """
    This trainer is to add VAT examples as one transformation of the image.
    MI(p,geom(p)) + MI(p,adv(p))
    """

    def _trainer_specific_loss(
        self,
        images: torch.Tensor,
        images_tf: torch.Tensor,
        pred: List[torch.Tensor],
        pred_tf: List[torch.Tensor],
    ) -> torch.Tensor:
        assert simplex(pred[0]) and pred_tf.__len__() == pred.__len__()

        # generate adversarial images: Take image without repetition
        _, adv_images, _ = VATLoss_Multihead(xi=1, eps=10, prop_eps=0.1)(
            self.model.torchnet, images[: self.train_loader.batch_size]
        )
        adv_preds = self.model(adv_images)
        assert adv_preds[0].__len__() == self.train_loader.batch_size

        batch_loss: List[torch.Tensor] = []  # type: ignore
        for subhead in range(pred.__len__()):
            # add adv prediction to the whole prediction list
            _loss, _loss_no_lambda = self.criterion(
                torch.cat(
                    (pred[subhead], pred[subhead][: self.train_loader.batch_size]),
                    dim=0,
                ),
                torch.cat((pred_tf[subhead], adv_preds[subhead]), dim=0),
            )
            batch_loss.append(_loss)
        batch_loss: torch.Tensor = sum(batch_loss) / len(batch_loss)  # type: ignore
        self.METERINTERFACE[f"train_mi"].add(-batch_loss.item())

        total_loss = batch_loss

        return total_loss


class IIC_adv_Trainer(IIC_Trainer):
    def _trainer_specific_loss(
        self,
        images: torch.Tensor,
        images_tf: torch.Tensor,
        pred: List[torch.Tensor],
        pred_tf: List[torch.Tensor],
    ) -> torch.Tensor:
        assert simplex(pred[0]) and pred_tf.__len__() == pred.__len__()
        _, adv_images, _ = VATLoss_Multihead(xi=1, eps=10, prop_eps=0.1)(
            self.model.torchnet, images
        )
        adv_pred = self.model(adv_images)
        assert simplex(adv_pred[0])

        batch_loss: List[torch.Tensor] = []  # type: ignore
        for subhead in range(pred.__len__()):
            _loss, _loss_no_lambda = self.criterion(pred[subhead], adv_pred[subhead])
            batch_loss.append(_loss)
        batch_loss: torch.Tensor = sum(batch_loss) / len(batch_loss)  # type: ignore
        self.METERINTERFACE[f"train_mi"].add(-batch_loss.item())
        return batch_loss


config = ConfigManger(DEFAULT_CONFIG_PATH="./config.yml", verbose=False).config

datainterface = MNISTClusteringDatasetInterface(
    split_partitions=["train"], **config["DataLoader"]
)
datainterface.drop_last = True
train_loader = datainterface.ParallelDataLoader(
    default_mnist_img_transform["tf1"],
    default_mnist_img_transform["tf2"],
    default_mnist_img_transform["tf2"],
    default_mnist_img_transform["tf2"],
    default_mnist_img_transform["tf2"],
    default_mnist_img_transform["tf2"],
)
datainterface.split_partitions = ["val"]
datainterface.drop_last = False
val_loader = datainterface.ParallelDataLoader(default_mnist_img_transform["tf3"])

model = Model(config["Arch"], config["Optim"], config["Scheduler"])

assert config["Trainer"]["name"] in (
    "IIC",
    "IIC_enhance",
    "IIC_adv_enhance",
    "IMSAT",
    "IMSAT_enhance",
)
if config["Trainer"]["name"] == "IMSAT":
    # MI(x,p) + CE(p,adv(p)) or MI(x,p) + CE(p,geom(p))
    Trainer = IMSAT_Trainer
elif config["Trainer"]["name"] == "IMSAT_enhance":
    # MI(x,p) + CE(p,adv(p)) + CE(p,geom(p))
    Trainer = IMSAT_Enhanced_Trainer
elif config["Trainer"]["name"] == "IIC":
    # MI(p,geom(p)) or  MI(p,geom(p))+kl(p, adv(p))
    Trainer = IIC_Trainer
elif config["Trainer"]["name"] == "IIC_enhance":
    # MI(p,geom(p)) + MI(p,adv(p))
    Trainer = IIC_enhanced_Trainer
elif config["Trainer"]["name"] == "IIC_adv_enhance":
    Trainer = IIC_adv_Trainer
else:
    raise NotImplementedError(config["Trainer"]["name"])

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    **config["Trainer"],
    config=config,
)
trainer.start_training()
trainer.clean_up()
