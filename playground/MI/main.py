from copy import deepcopy as dcp
from typing import List

import matplotlib
import torch
from torch.utils.data import DataLoader

from deepclustering import ModelMode
from deepclustering.dataset.segmentation.toydataset import Cls_ShapesDataset
from deepclustering.loss.IMSAT_loss import MultualInformaton_IMSAT
from deepclustering.manager import ConfigManger
from deepclustering.meters import MeterInterface, AverageValueMeter
from deepclustering.model import Model
from deepclustering.trainer.Trainer import _Trainer
from deepclustering.utils import tqdm_, simplex, tqdm, flatten_dict
from deepclustering.utils.classification.assignment_mapping import hungarian_match, flat_acc

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import pandas as pd


class Trainer(_Trainer):
    def __init__(self, model: Model, train_loader: DataLoader, val_loader: DataLoader, max_epoch: int = 100,
                 save_dir: str = 'MI', checkpoint_path: str = None, device='cpu', config: dict = None) -> None:
        super().__init__(model, train_loader, val_loader, max_epoch, save_dir, checkpoint_path, device, config)
        self.criterion = MultualInformaton_IMSAT(mu=4, verbose=True)
        plt.ion()

    def __init_meters__(self) -> List[str]:
        METER_CONFIG = {
            'train_mi': AverageValueMeter(),
            'val_best_acc': AverageValueMeter(),
            'val_average_acc': AverageValueMeter()
        }
        self.METERINTERFACE = MeterInterface(METER_CONFIG)
        return ["train_mi_mean", 'val_average_acc_mean', "val_best_acc_mean"]

    def _train_loop(self, train_loader: DataLoader, epoch: int, mode=ModelMode.TRAIN, *args, **kwargs):
        super()._train_loop(*args, **kwargs)
        self.model.set_mode(mode)
        assert self.model.training
        _train_loader: tqdm = tqdm_(train_loader)
        for _batch_num, (img, _) in enumerate(_train_loader):
            img = img.to(self.device)
            pred = self.model(img)
            assert simplex(pred[0]), pred
            _losses = []
            for subhead_num in range(self.model.arch_dict['num_sub_heads']):
                _losses.append(-self.criterion(pred[subhead_num]))
            loss = sum(_losses) / len(_losses)
            self.METERINTERFACE['train_mi'].add(-loss.item())
            self.model.zero_grad()
            loss.backward()
            self.model.step()

            report_dict = flatten_dict({'train_MI': self.METERINTERFACE.train_mi.summary()})
            _train_loader.set_postfix(report_dict)

        report_dict_str = ', '.join([f'{k}:{v:.3f}' for k, v in report_dict.items()])
        print(f"  Training epoch: {epoch} : {report_dict_str}")

    def _eval_loop(self, val_loader: DataLoader, epoch: int, mode=ModelMode.EVAL, *args, **kwargs) -> float:
        super(Trainer, self)._eval_loop(*args, **kwargs)
        self.model.set_mode(mode)
        assert not self.model.training
        _val_loader = tqdm_(val_loader)
        preds = torch.zeros(self.model.arch_dict['num_sub_heads'],
                            val_loader.dataset.__len__(),
                            dtype=torch.long,
                            device=self.device)
        probas = torch.zeros(self.model.arch_dict['num_sub_heads'],
                             val_loader.dataset.__len__(),
                             self.model.arch_dict['output_k'],
                             dtype=torch.float,
                             device=self.device)
        gts = torch.zeros(val_loader.dataset.__len__(), dtype=torch.long, device=self.device)
        _batch_done = 0
        for _batch_num, (img, gt) in enumerate(_val_loader):
            img, gt = img.to(self.device), gt.to(self.device)
            pred = self.model(img)
            _bSlice = slice(_batch_done, _batch_done + img.shape[0])
            gts[_bSlice] = gt
            for subhead in range(pred.__len__()):
                preds[subhead][_bSlice] = pred[subhead].max(1)[1]
                probas[subhead][_bSlice] = pred[subhead]
            _batch_done += img.shape[0]
        assert _batch_done == val_loader.dataset.__len__(), _batch_done

        # record
        subhead_accs = []
        for subhead in range(self.model.arch_dict['num_sub_heads']):
            reorder_pred, remap = hungarian_match(
                flat_preds=preds[subhead],
                flat_targets=gts,
                preds_k=self.model.arch_dict['output_k'],
                targets_k=self.model.arch_dict['output_k']
            )
            _acc = flat_acc(reorder_pred, gts)
            subhead_accs.append(_acc)
            # record average acc
            self.METERINTERFACE.val_average_acc.add(_acc)
        self.METERINTERFACE.val_best_acc.add(max(subhead_accs))
        report_dict = {'average_acc': self.METERINTERFACE.val_average_acc.summary()['mean'],
                       'best_acc': self.METERINTERFACE.val_best_acc.summary()['mean']}
        report_dict_str = ', '.join([f'{k}:{v:.3f}' for k, v in report_dict.items()])
        print(f"Validating epoch: {epoch} : {report_dict_str}")
        self.writer.add_histogram(tag='probas', values=probas[0], global_step=epoch)
        # plt.clf()
        # plt.imshow((probas.squeeze().unsqueeze(1) * probas.squeeze().unsqueeze(2)).mean(0).cpu().detach())
        # plt.colorbar()
        # plt.show()
        # plt.pause(0.001)
        plt.clf()
        probas = pd.DataFrame(probas.squeeze().cpu().numpy())
        probas[0].plot.density(label='0')
        probas[1].plot.density(label='1')
        probas[2].plot.density(label='2')
        plt.legend()
        plt.grid()
        plt.show()
        plt.pause(0.001)
        print(pd.Series(preds.cpu().numpy().ravel()).value_counts())

        return self.METERINTERFACE.val_best_acc.summary()['mean']


config = ConfigManger(DEFAULT_CONFIG_PATH='./config.yml', verbose=True).config

trainset = Cls_ShapesDataset(height=128, width=128, max_object_scale=0.75, )
valset = dcp(trainset)
train_loader = DataLoader(trainset, batch_size=20, shuffle=True, pin_memory=True)
val_loader = DataLoader(valset, batch_size=20, shuffle=False, pin_memory=True)

model = Model(config['Arch'], config['Optim'], config['Scheduler'])

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    **config['Trainer'],
    config=config
)
trainer.start_training()
