from typing import List

from torch.utils.data import DataLoader

from deepclustering.loss.IMSAT_loss import Perturbation_Loss, MultualInformaton_IMSAT
from deepclustering.meters import AverageValueMeter, MeterInterface
from deepclustering.model import Model
from deepclustering.trainer import IICMultiHeadTrainer
from deepclustering.utils import dict_filter
from deepclustering.utils.VAT import VATLoss_Multihead


class IMSATIICTrainer(IICMultiHeadTrainer):
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
        adv_weight=0.1,
    ) -> None:
        super().__init__(
            model,
            train_loader_A,
            train_loader_B,
            val_loader,
            max_epoch,
            save_dir,
            checkpoint_path,
            device,
            head_control_params,
            use_sobel,
            config,
        )

        self.p_criterion = Perturbation_Loss()  # kl_div
        self.MI = MultualInformaton_IMSAT()  # mutual information
        self.adv_weight = float(adv_weight)

    def __init_meters__(self) -> List[str]:
        METER_CONFIG = {
            "train_head_A": AverageValueMeter(),
            "train_head_B": AverageValueMeter(),
            "train_adv_A": AverageValueMeter(),
            "train_adv_B": AverageValueMeter(),
            "val_average_acc": AverageValueMeter(),
            "val_best_acc": AverageValueMeter(),
        }
        self.METERINTERFACE = MeterInterface(METER_CONFIG)
        return [
            "train_head_A_mean",
            "train_head_B_mean",
            "train_adv_A_mean",
            "train_adv_B_mean",
            "val_average_acc_mean",
            "val_best_acc_mean",
        ]

    @property
    def _training_report_dict(self):
        report_dict = {
            "train_MI_A": self.METERINTERFACE["train_head_A"].summary()["mean"],
            "train_MI_B": self.METERINTERFACE["train_head_B"].summary()["mean"],
            "train_adv_A": self.METERINTERFACE["train_adv_A"].summary()["mean"],
            "train_adv_B": self.METERINTERFACE["train_adv_B"].summary()["mean"],
        }
        report_dict = dict_filter(report_dict, lambda k, v: v != 0.0)
        return report_dict

    @property
    def _eval_report_dict(self):
        report_dict = {
            "average_acc": self.METERINTERFACE.val_average_acc.summary()["mean"],
            "best_acc": self.METERINTERFACE.val_best_acc.summary()["mean"],
        }
        report_dict = dict_filter(report_dict, lambda k, v: v != 0.0)
        return report_dict

    def _trainer_specific_loss(self, tf1_images, tf2_images, head_name):
        iic_loss = super()._trainer_specific_loss(tf1_images, tf2_images, head_name)
        adv_loss = 0
        if self.adv_weight > 0:
            adv_loss, *_ = VATLoss_Multihead(xi=0.25, eps=1, prop_eps=0.1)(
                self.model.torchnet, tf1_images
            )
            self.METERINTERFACE[f"train_adv_{head_name}"].add(adv_loss.item())
        total_loss = (
            iic_loss + self.adv_weight * adv_loss
        )  # here the mi_batch_loss shoud be negative
        return total_loss
