"""
This is the trainer for IIC multiple-header Clustering
"""
from torch.utils.data import DataLoader
from .. import ModelMode
from ..meters import AverageValueMeter, MeterInterface
from ..model import Model
from ..utils import tqdm_, simplex, tqdm
from .trainer import _Trainer
from ..loss.IID_losses import IIDLoss
from ..utils.classification.assignment_mapping import flat_acc, hungarian_match
import torch
from collections import OrderedDict


class IICMultiHeadTrainer(_Trainer):
    METER_CONFIG = {
        'train_head_A': AverageValueMeter(),
        'train_head_B': AverageValueMeter(),
        'val_average_acc': AverageValueMeter(),
        'val_best_acc': AverageValueMeter()
    }
    METERINTERFACE = MeterInterface(METER_CONFIG)

    def __init__(
            self,
            model: Model,
            train_loader_A: DataLoader,
            train_loader_B: DataLoader,
            val_loader: DataLoader,
            max_epoch: int = 100,
            save_dir: str = './runs/IICMultiHead',
            checkpoint_path: str = None,
            device='cpu',
            head_control_params: dict = {'A': 1, 'B': 2}
    ) -> None:
        super().__init__(model, None, val_loader, max_epoch, save_dir, checkpoint_path, device)
        self.train_loader_A = train_loader_A
        self.train_loader_B = train_loader_B
        assert self.train_loader is None
        self.head_control_params: OrderedDict = OrderedDict(head_control_params)
        self.criterion = IIDLoss()
        self.criterion.to(self.device)

    def start_training(self):
        for epoch in range(self.max_epoch):
            self._train_loop(
                train_loader_A=self.train_loader_A,
                train_loader_B=self.train_loader_B,
                epoch=epoch,
                head_control_param=self.head_control_params
            )
            with torch.no_grad():
                self._eval_loop(self.val_loader, epoch)
            self.METERINTERFACE.step()

            print(self.METERINTERFACE.summary())

    def _train_loop(
            self,
            train_loader_A: DataLoader,
            train_loader_B: DataLoader,
            epoch: int,
            mode: ModelMode = ModelMode.TRAIN,
            head_control_param={},
            *args,
            **kwargs
    ):
        """
        :param train_loader: TrainLoader which is the same as the Val
        :param epoch: current epoch
        :param mode: should be ModelMode.TRAIN
        :param args: to ignore
        :param kwargs: to ignore
        :return:
        """
        assert head_control_param.__len__() > 0, f"`head_control_param` must be provided, given {head_control_param}."
        assert set(head_control_param.keys()) <= {'A', 'B'}, f"`head_control_param` key must be in `A` or `B`," \
            f" given{set(head_control_param.keys())}"
        for k, v in head_control_param.items():
            assert k in ('A', 'B'), f"`head_control_param` key must be in `A` or `B`," \
                f" given{set(head_control_param.keys())}"
            assert isinstance(v, int) and v > 0, f"Iteration for {k} must be > 0."
        self.model.set_mode(mode)
        assert self.model.training, f"Model should be in train() model, given {self.model.training}."
        assert len(train_loader_B) == len(train_loader_A), f"The length of the train_loaders should be the same,\"" \
            f"given `len(train_loader_A)`:{len(train_loader_A)} and `len(train_loader_B)`:{len(train_loader_B)}."
        for head_name, head_iterations in head_control_param.items():
            assert head_name in ('A', 'B'), head_name
            train_loader = eval(f"train_loader_{head_name}")  # change the dataloader for different head
            for head_epoch in range(head_iterations):
                # given one head, one iteration in this head, and one train_loader.
                train_loader_: tqdm = tqdm_(train_loader)  # reinitilize the train_loader
                train_loader_.set_description(
                    f'Training epoch: {epoch} head:{head_name}, head_epoch:{head_epoch + 1}/{head_iterations}')
                for batch, image_labels in enumerate(train_loader_):
                    images, _ = list(zip(*image_labels))
                    tf1_images = torch.cat([images[0] for _ in range(images.__len__() - 1)], dim=0).to(self.device)
                    tf2_images = torch.cat(images[1:], dim=0).to(self.device)
                    assert tf1_images.shape == tf2_images.shape
                    self.model.zero_grad()
                    tf1_pred_simplex = self.model.torchnet(tf1_images, head=head_name)
                    tf2_pred_simplex = self.model.torchnet(tf2_images, head=head_name)
                    assert simplex(tf1_pred_simplex[0]) and tf1_pred_simplex.__len__() == tf2_pred_simplex.__len__()
                    batch_loss = []
                    for subhead in range(tf1_pred_simplex.__len__()):
                        _loss, _loss_no_lambda = self.criterion(tf1_pred_simplex[subhead], tf2_pred_simplex[subhead])
                        batch_loss.append(_loss)
                    batch_loss = sum(batch_loss) / len(batch_loss)
                    self.METERINTERFACE[f'train_head_{head_name}'].add(-batch_loss.item())
                    self.model.zero_grad()
                    batch_loss.backward()
                    self.model.step()
                    train_loader_.set_postfix(self.METERINTERFACE[f'train_head_{head_name}'].summary())
        report_dict = {'train_head_A': self.METERINTERFACE['train_head_A'].summary()['mean'],
                       'train_head_B': self.METERINTERFACE['train_head_B'].summary()['mean']}
        report_dict_str = ', '.join([f'{k}:{v:.3f}' for k, v in report_dict.items()])
        print(f"Training epoch: {epoch} : {report_dict_str}")

    def _eval_loop(self, val_loader: DataLoader, epoch: int, mode: ModelMode = ModelMode.EVAL, *args, **kwargs):
        self.model.set_mode(mode)
        assert not self.model.training, f"Model should be in eval model in _eval_loop, given {self.model.training}."
        val_loader_: tqdm = tqdm_(val_loader)
        preds = torch.zeros(self.model.arch_dict['num_sub_heads'],
                            val_loader.dataset.__len__(),
                            dtype=torch.long,
                            device=self.device)
        target = torch.zeros(val_loader.dataset.__len__(),
                             dtype=torch.long,
                             device=self.device)

        slice_done = 0
        subhead_accs = []
        val_loader_.set_description(f'Validating epoch: {epoch}')
        for batch, image_labels in enumerate(val_loader_):
            images, gt = list(zip(*image_labels))
            images, gt = images[0].to(self.device), gt[0].to(self.device)
            _pred = self.model.torchnet(images, head='B')
            assert _pred.__len__() == self.model.arch_dict['num_sub_heads']
            assert simplex(_pred[0]), f"pred should be normalized, given {_pred[0]}."
            bSlicer = slice(slice_done, slice_done + images.shape[0])
            for subhead in range(self.model.arch_dict['num_sub_heads']):
                preds[subhead][bSlicer] = _pred[subhead].max(1)[1]
                target[bSlicer] = gt

            slice_done += gt.shape[0]
        assert slice_done == val_loader.dataset.__len__(), 'Slice not completed.'
        for subhead in range(self.model.arch_dict['num_sub_heads']):
            reorder_pred, remap = hungarian_match(
                flat_preds=preds[subhead],
                flat_targets=target,
                preds_k=10,
                targets_k=10
            )
            _acc = flat_acc(reorder_pred, target)
            subhead_accs.append(_acc)
            # record average acc
            self.METERINTERFACE.val_average_acc.add(_acc)

        # record best acc
        self.METERINTERFACE.val_best_acc.add(max(subhead_accs))
        report_dict = {'best_acc': self.METERINTERFACE.val_average_acc.summary()['mean'],
                       'train_head_B': self.METERINTERFACE.val_best_acc.summary()['mean']}
        report_dict_str = ', '.join([f'{k}:{v:.3f}' for k, v in report_dict.items()])
        print(f"Validating epoch: {epoch} : {report_dict_str}")

    @property
    def state_dict(self):
        return None
