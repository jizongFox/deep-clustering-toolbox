"""
This is to investigate the GN and BN
"""
from typing import List

import torch
from resnet import resnet101
from analyse_model import analyze_alpha
from torch import nn
# dataset
from torch.utils.data import DataLoader

from deepclustering import ModelMode, DATA_PATH, RUNS_PATH
from deepclustering import optim
from deepclustering.dataset.classification.cifar import CIFAR10
from deepclustering.dataset.classification.cifar_helper import \
    default_cifar10_img_transform
from deepclustering.dataset.dataset import CombineDataset
from deepclustering.meters import AverageValueMeter, MeterInterface
from deepclustering.model import Model
from deepclustering.trainer.Trainer import _Trainer
from deepclustering.utils import yaml_load, tqdm_

for k, v in default_cifar10_img_transform.items():
    v.transforms[-1].include_rgb = True
    v.transforms[-1].include_grey = False

default_config = yaml_load('resnet.yaml', verbose=False)

dataloader_dict = default_config['DataLoader']
train_set_list = [CIFAR10(root=DATA_PATH, train=True, download=True,
                          transform=default_cifar10_img_transform[t]) for t in ('tf1', 'tf2', 'tf2', 'tf2', 'tf2')]
train_loader = DataLoader(CombineDataset(*train_set_list), **dataloader_dict)
val_loader = DataLoader(
    CombineDataset(*[CIFAR10(root=DATA_PATH,
                             train=False,
                             transform=default_cifar10_img_transform['tf3'],
                             download=True)]),
    **dataloader_dict)

# network
model = Model(default_config['Arch'], default_config['Optim'], default_config['Scheduler'])
net: nn.Module = resnet101(num_classes=10)
optimizer = optim.Adam(net.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=default_config['Scheduler']['milestones'])
model.torchnet = net
model.optimizer = optimizer
model.scheduler = scheduler


class class_Trainer(_Trainer):

    def __init__(self, model: Model, train_loader: DataLoader, val_loader: DataLoader, max_epoch: int = 1000,
                 save_dir: str = RUNS_PATH + "/gn_bn", checkpoint_path: str = None, device='cpu',
                 config: dict = None) -> None:
        super().__init__(model, train_loader, val_loader, max_epoch, save_dir, checkpoint_path, device, config)
        self.criterion = nn.CrossEntropyLoss()

    def __init_meters__(self) -> List[str]:
        METER_CONFIG = {
            'train_loss': AverageValueMeter(),
            'train_acc': AverageValueMeter(),
            'val_acc': AverageValueMeter(),
            'val_loss': AverageValueMeter()
        }
        self.METERINTERFACE = MeterInterface(METER_CONFIG)
        return ['train_loss_mean',
                'val_loss_mean',
                'train_acc_mean',
                'val_acc_mean']

    def _train_loop(self, train_loader, epoch, mode=ModelMode.TRAIN, **kwargs):
        self.model.set_mode(mode)
        assert self.model.training

        train_loader_ = tqdm_(train_loader)
        for batch, image_labels in enumerate(train_loader_):
            images, labels = list(zip(*image_labels))
            # print(f"used time for dataloading:{time.time() - time_before}")
            tf1_images = torch.cat([images[0] for _ in range(images.__len__() - 1)], dim=0).to(self.device)
            tf2_images = torch.cat(images[1:], dim=0).to(self.device)
            labels = torch.cat([labels[0] for _ in range(labels.__len__() - 1)], dim=0).to(self.device)
            assert tf1_images.shape == tf2_images.shape
            tf1_pred_logit = self.model.torchnet(tf1_images)
            tf2_pred_logit = self.model.torchnet(tf2_images)
            loss1 = self.criterion(tf1_pred_logit, labels)
            loss2 = self.criterion(tf2_pred_logit, labels)
            batch_loss = loss1 + loss2
            _acc = (tf1_pred_logit.max(1)[1] == labels).float().sum() / float(len(labels))
            self.METERINTERFACE[f'train_loss'].add(batch_loss.item())
            self.METERINTERFACE[f'train_acc'].add(_acc.item())
            self.model.zero_grad()
            batch_loss.backward()
            self.model.step()
            report_dict = {'train_loss': self.METERINTERFACE[f'train_loss'].summary()['mean'],
                           'train_acc': self.METERINTERFACE['train_acc'].summary()['mean']}
            train_loader_.set_postfix(report_dict)

        report_dict_str = ', '.join([f'{k}:{v:.3f}' for k, v in report_dict.items()])
        print(f"Training epoch: {epoch} : {report_dict_str}")

    def _eval_loop(self, val_loader, epoch, mode=ModelMode.EVAL, **kwargs) -> float:
        self.model.set_mode(mode)
        assert not self.model.training

        val_loader_ = tqdm_(val_loader)
        for batch, image_labels in enumerate(val_loader_):
            images, labels = list(zip(*image_labels))
            images = images[0].to(self.device)
            labels = labels[0].to(self.device)
            _pred_logit = self.model.torchnet(images)
            loss1 = self.criterion(_pred_logit, labels)
            batch_loss = loss1
            _acc = (_pred_logit.max(1)[1] == labels).float().sum() / float(len(labels))
            self.METERINTERFACE[f'val_loss'].add(batch_loss.item())
            self.METERINTERFACE[f'val_acc'].add(_acc.item())
            report_dict = {'val_loss': self.METERINTERFACE[f'val_loss'].summary()['mean'],
                           'val_acc': self.METERINTERFACE['val_acc'].summary()['mean']}
            val_loader_.set_postfix(report_dict)
        report_dict_str = ', '.join([f'{k}:{v:.3f}' for k, v in report_dict.items()])
        print(f"Validating epoch: {epoch} : {report_dict_str}")
        return self.METERINTERFACE['val_acc'].summary()['mean']

    def start_training(self):
        for epoch in range(self._start_epoch, self.max_epoch):
            self._train_loop(
                train_loader=self.train_loader,
                epoch=epoch,
            )
            with torch.no_grad():
                current_score = self._eval_loop(self.val_loader, epoch)
            self.METERINTERFACE.step()
            self.model.schedulerStep()
            # save meters and checkpoints
            for k, v in self.METERINTERFACE.aggregated_meter_dict.items():
                v.summary().to_csv(self.save_dir / f'meters/{k}.csv')
            self.METERINTERFACE.summary().to_csv(self.save_dir / f'wholeMeter.csv')
            self.writer.add_scalars('Scalars', self.METERINTERFACE.summary().iloc[-1].to_dict(), global_step=epoch)
            self.drawer.draw(self.METERINTERFACE.summary(), together=False)
            self.save_checkpoint(self.state_dict, epoch, current_score)
            if epoch % 10 == 0:
                analyze_alpha(model_cls=resnet101, checkpoint_path=self.save_dir / 'last.pth', epoch=epoch,
                              save_dir=self.save_dir / 'alphas')


trainer = class_Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    **default_config['Trainer'],
    config=default_config

)
trainer.start_training()
