"""
This is to investigate the GN and BN
"""
import torch
from deepclustering import ModelMode
from deepclustering import optim
from deepclustering.dataset.classification.cifar_helper import \
    Cifar10ClusteringDataloaders, default_cifar10_img_transform
from deepclustering.model import Model
from deepclustering.trainer.Trainer import _Trainer
from deepclustering.utils import yaml_load, tqdm_, simplex
from resnet import resnet18
from torch import nn
# dataset
from torch.utils.data import DataLoader

dataloader_generator = Cifar10ClusteringDataloaders(batch_size=100, shuffle=True, num_workers=4, pin_memory=True)
train_loader = dataloader_generator.creat_CombineDataLoader(
    default_cifar10_img_transform['tf1'],
    default_cifar10_img_transform['tf2'],
    default_cifar10_img_transform['tf2']
)
val_loader = dataloader_generator.creat_CombineDataLoader(
    default_cifar10_img_transform['tf3']
)
# network
default_config = yaml_load('resnet.yaml', verbose=False)
model = Model(default_config['Arch'], default_config['Optim'], default_config['Scheduler'])
net: nn.Module = resnet18(num_classes=10)
optimizer = optim.Adam(net.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=default_config['Scheduler']['milestones'])
model.torchnet = net
model.optimizer = optimizer
model.scheduler = scheduler


class class_Trainer(_Trainer):
    METER_CONFIG = {}
    METERINTERFACE = None

    def __init__(self, model: Model, train_loader: DataLoader, val_loader: DataLoader, max_epoch: int = 100,
                 save_dir: str = './runs/test', checkpoint_path: str = None, device='cpu', config: dict = None) -> None:
        super().__init__(model, train_loader, val_loader, max_epoch, save_dir, checkpoint_path, device, config)
        self.criterion = nn.CrossEntropyLoss()

    def _train_loop(self, train_loader, epoch, mode=ModelMode.TRAIN, **kwargs):
        self.model.set_mode(mode)
        assert self.model.training

        train_loader_ = tqdm_(train_loader)
        for batch, image_labels in enumerate(train_loader_):
            images, _ = list(zip(*image_labels))
            # print(f"used time for dataloading:{time.time() - time_before}")
            tf1_images = torch.cat([images[0] for _ in range(images.__len__() - 1)], dim=0).to(self.device)
            tf2_images = torch.cat(images[1:], dim=0).to(self.device)
            assert tf1_images.shape == tf2_images.shape
            tf1_pred_simplex = self.model.torchnet(tf1_images)
            tf2_pred_simplex = self.model.torchnet(tf2_images)
            assert simplex(tf1_pred_simplex[0]) and tf1_pred_simplex.__len__() == tf2_pred_simplex.__len__()
            batch_loss = []
            for subhead in range(tf1_pred_simplex.__len__()):
                _loss, _loss_no_lambda = self.criterion(tf1_pred_simplex[subhead], tf2_pred_simplex[subhead])
                batch_loss.append(_loss)
            batch_loss: torch.Tensor = sum(batch_loss) / len(batch_loss)
            self.METERINTERFACE[f'train_head_{head_name}'].add(-batch_loss.item())
            self.model.zero_grad()
            batch_loss.backward()
            self.model.step()
            train_loader_.set_postfix(self.METERINTERFACE[f'train_head_{head_name}'].summary())
            # time_before = time.time()
        report_dict = {'train_head_A': self.METERINTERFACE['train_head_A'].summary()['mean'],
                       'train_head_B': self.METERINTERFACE['train_head_B'].summary()['mean']}
        report_dict_str = ', '.join([f'{k}:{v:.3f}' for k, v in report_dict.items()])
        print(f"Training epoch: {epoch} : {report_dict_str}")


trainer = class_Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    save_dir="./runs/bn_gn",
    device='cuda'
)
trainer.start_training()