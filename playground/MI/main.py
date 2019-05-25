import warnings
from typing import List, Callable

import matplotlib
import torch
from torch.utils.data import DataLoader

from deepclustering import ModelMode
from deepclustering.dataset.classification.clustering_helper import ClusterDatasetInterface
from deepclustering.dataset.segmentation.toydataset import Cls_ShapesDataset, default_toy_img_transform
from deepclustering.loss.IMSAT_loss import MultualInformaton_IMSAT
from deepclustering.loss.loss import JSD
from deepclustering.manager import ConfigManger
from deepclustering.meters import MeterInterface, AverageValueMeter
from deepclustering.model import Model
from deepclustering.trainer.Trainer import _Trainer
from deepclustering.utils import tqdm_, simplex, tqdm, flatten_dict
from deepclustering.utils.classification.assignment_mapping import hungarian_match, flat_acc

matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings('ignore')


class ToyExampleInterFace(ClusterDatasetInterface):
    ALLOWED_SPLIT = ['1']

    def __init__(self, batch_size: int = 1, shuffle: bool = False,
                 num_workers: int = 1, pin_memory: bool = True, drop_last=False) -> None:
        super().__init__(Cls_ShapesDataset, ['1'], batch_size, shuffle, num_workers, pin_memory, drop_last)

    def _creat_concatDataset(self, image_transform: Callable, target_transform: Callable, dataset_dict: dict = {}):
        train_set = Cls_ShapesDataset(
            count=5000,
            height=100,
            width=100,
            max_object_scale=0.75,
            transform=image_transform,
            target_transform=target_transform,
            **dataset_dict)
        return train_set


class Trainer(_Trainer):
    def __init__(self, model: Model, train_loader: DataLoader, val_loader: DataLoader, max_epoch: int = 100,
                 save_dir: str = 'MI', checkpoint_path: str = None, device='cpu', config: dict = None) -> None:
        super().__init__(model, train_loader, val_loader, max_epoch, save_dir, checkpoint_path, device, config)
        self.criterion = MultualInformaton_IMSAT(mu=4, separate_return=True)
        self.jsd = JSD()
        plt.ion()

    def __init_meters__(self) -> List[str]:
        METER_CONFIG = {
            'train_mi': AverageValueMeter(),
            'train_entropy': AverageValueMeter(),
            'train_centropy': AverageValueMeter(),
            'train_sat': AverageValueMeter(),
            'val_best_acc': AverageValueMeter(),
            'val_average_acc': AverageValueMeter()
        }
        self.METERINTERFACE = MeterInterface(METER_CONFIG)
        return [
            "train_mi_mean",
            'train_entropy_mean',
            'train_centropy_mean',
            'train_sat_mean',
            'val_average_acc_mean',
            "val_best_acc_mean"
        ]

    def _train_loop(self, train_loader: DataLoader, epoch: int, mode=ModelMode.TRAIN, *args, **kwargs):
        super()._train_loop(*args, **kwargs)
        self.model.set_mode(mode)
        assert self.model.training
        _train_loader: tqdm = tqdm_(train_loader)
        for _batch_num, (img_labels) in enumerate(_train_loader):
            images, labels = zip(*img_labels)
            tf1_images = torch.cat(tuple([images[0] for _ in range(images.__len__() - 1)]), dim=0).to(self.device)
            tf2_images = torch.cat(tuple(images[1:]), dim=0).to(self.device)
            pred_tf1_simplex = self.model(tf1_images)
            pred_tf2_simplex = self.model(tf2_images)
            assert simplex(pred_tf1_simplex[0]), pred_tf1_simplex
            mi_losses, entropy_losses, centropy_losses = [], [], []
            for subhead_num in range(self.model.arch_dict['num_sub_heads']):
                _mi_loss, (_entropy_loss, _centropy_loss) = self.criterion(pred_tf1_simplex[subhead_num])
                mi_losses.append(-_mi_loss)
                entropy_losses.append(_entropy_loss)
                centropy_losses.append(_centropy_loss)
            mi_loss = sum(mi_losses) / len(mi_losses)
            entrop_loss = sum(entropy_losses) / len(entropy_losses)
            centropy_loss = sum(centropy_losses) / len(centropy_losses)

            self.METERINTERFACE['train_mi'].add(-mi_loss.item())
            self.METERINTERFACE['train_entropy'].add(entrop_loss.item())
            self.METERINTERFACE['train_centropy'].add(centropy_loss.item())

            _sat_loss = list(map(lambda p1, p2: self.jsd([p2, p1]), pred_tf1_simplex, pred_tf2_simplex))
            sat_loss = sum(_sat_loss) / len(_sat_loss)
            self.METERINTERFACE['train_sat'].add(sat_loss.item())

            total_loss = mi_loss + 0.1 * sat_loss
            self.model.zero_grad()
            total_loss.backward()
            self.model.step()

            report_dict = flatten_dict({'train_MI': self.METERINTERFACE.train_mi.summary(),
                                        'train_entropy': self.METERINTERFACE.train_entropy.summary(),
                                        'train_centropy': self.METERINTERFACE.train_centropy.summary(),
                                        'train_sat': self.METERINTERFACE.train_sat.summary()})
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
        for _batch_num, (img_labels) in enumerate(_val_loader):
            images, labels = zip(*img_labels)
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

datainterface = ToyExampleInterFace(**config['DataLoader'])
train_loader = datainterface.ParallelDataLoader(
    default_toy_img_transform['tf1']['img'],
    default_toy_img_transform['tf2']['img'],
    default_toy_img_transform['tf2']['img'],
    default_toy_img_transform['tf2']['img'],

)
val_loader = datainterface.ParallelDataLoader(
    default_toy_img_transform['tf3']['img'],
    target_transform=[default_toy_img_transform['tf3']['target']]
)

model = Model(config['Arch'], config['Optim'], config['Scheduler'])

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    **config['Trainer'],
    config=config
)
trainer.start_training()
