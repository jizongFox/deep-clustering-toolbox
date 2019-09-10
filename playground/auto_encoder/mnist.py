from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from deepclustering import DATA_PATH, ModelMode
from deepclustering.manager import ConfigManger
from deepclustering.meters import MeterInterface, AverageValueMeter
from deepclustering.model import Model
from deepclustering.trainer import _Trainer
from deepclustering.utils import tqdm_, tqdm


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class MNISTTrainer(_Trainer):
    def __init__(
        self,
        model: Model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        max_epoch: int = 100,
        save_dir: str = "mnist",
        checkpoint_path: str = None,
        device="cpu",
        config: dict = None,
        **kwargs
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
            **kwargs
        )
        self.criterion = criterion

    def __init_meters__(self) -> List[str]:
        METER_CONFIG = {"rec_loss": AverageValueMeter()}
        self.METERINTERFACE = MeterInterface(METER_CONFIG)
        return ["rec_loss_mean"]

    def _training_report_dict(self):
        return {"rec_loss": self.METERINTERFACE["rec_loss"].summary()["mean"]}

    def _train_loop(
        self, train_loader=None, epoch=0, mode=ModelMode.TRAIN, *args, **kwargs
    ):
        self.model.train()
        train_loader_: tqdm = tqdm_(train_loader)
        for batch_num, data in enumerate(train_loader_):
            img, _ = data
            img = img.to(self.device)
            # ===================forward=====================
            output = self.model(img)
            loss = self.criterion(output, img)
            # ===================backward====================
            self.model.zero_grad()
            loss.backward()
            self.model.step()
            self.METERINTERFACE.rec_loss.add(loss.item())
            train_loader_.set_postfix(self._training_report_dict())

    def _eval_report_dict(self):
        pass

    def _eval_loop(
        self,
        val_loader: DataLoader = None,
        epoch: int = 0,
        mode=ModelMode.EVAL,
        *args,
        **kwargs
    ) -> float:
        from torchvision.utils import make_grid

        # self.model.eval()
        for batch_num, data in enumerate(val_loader):
            img, _ = data
            img = img.to(self.device)
            # ===================forward=====================
            output = self.model(img)
            output_draw = make_grid((output + 1) / 2)
            self.writer.add_image("rec", output_draw, global_step=epoch)
            break

        return epoch * 0.001


class gradient_difference_loss(nn.Module):
    def __init__(self, gdl_weight=0.01):
        super().__init__()
        self.gdl_weight = float(gdl_weight)
        self.mse = nn.MSELoss()
        self.abs_loss = lambda x, y: F.l1_loss(x, y).mean()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        assert pred.max() <= 1 and pred.min() >= 0
        assert gt.max() == 1 and gt.min() == 0
        orig_gradient_x = torch.roll(gt, dims=2, shifts=1) - gt
        pred_gradient_x = torch.roll(pred, dims=2, shifts=1) - pred
        orig_gradient_y = torch.roll(gt, dims=3, shifts=1) - gt
        pred_gradient_y = torch.roll(pred, dims=3, shifts=1) - pred
        gdl_loss = self.abs_loss(orig_gradient_x, pred_gradient_x) + self.abs_loss(
            orig_gradient_y, pred_gradient_y
        )
        return self.mse(pred, gt) + self.gdl_weight * gdl_loss


img_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize((0.5), (0.5))
    ]
)
dataset = MNIST(DATA_PATH, transform=img_transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

model = Model()
model.torchnet = autoencoder()
model.optimizer = torch.optim.Adam(
    model.torchnet.parameters(), lr=1e-3, weight_decay=1e-5
)

config = ConfigManger().parsed_args
if config["loss"] == "mse":
    criterion = nn.MSELoss()
elif config["loss"] == "gdl":
    criterion = gradient_difference_loss(config["weight"])

trainer = MNISTTrainer(
    model=model,
    train_loader=dataloader,
    val_loader=dataloader,
    criterion=nn.MSELoss(),
    device="cuda",
    **config["Trainer"]
)
trainer.start_training()
