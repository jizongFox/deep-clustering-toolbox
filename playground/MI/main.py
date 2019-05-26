import warnings
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
from deepclustering import ModelMode
from deepclustering.dataset.classification.mnist_helper import MNISTDatasetInterface, default_mnist_img_transform
# from deepclustering.dataset.segmentation.toydataset import Cls_ShapesDataset, default_toy_img_transform
from deepclustering.loss.IMSAT_loss import MultualInformaton_IMSAT
from deepclustering.loss.loss import JSD, KL_div
from deepclustering.manager import ConfigManger
from deepclustering.meters import MeterInterface, AverageValueMeter
from deepclustering.model import Model
from deepclustering.trainer.Trainer import _Trainer
from deepclustering.utils import tqdm_, simplex, tqdm, flatten_dict
from deepclustering.utils.VAT import VATLoss, _l2_normalize, _kl_div, _disable_tracking_bn_stats
from deepclustering.utils.classification.assignment_mapping import hungarian_match, flat_acc
from torch import nn
from torch.utils.data import DataLoader

matplotlib.use('tkagg')
warnings.filterwarnings('ignore')


# class ToyExampleInterFace(ClusterDatasetInterface):
#     ALLOWED_SPLIT = ['1']
#
#     def __init__(self, batch_size: int = 1, shuffle: bool = False,
#                  num_workers: int = 1, pin_memory: bool = True, drop_last=False) -> None:
#         super().__init__(Cls_ShapesDataset, ['1'], batch_size, shuffle, num_workers, pin_memory, drop_last)
#
#     def _creat_concatDataset(self, image_transform: Callable, target_transform: Callable, dataset_dict: dict = {}):
#         train_set = Cls_ShapesDataset(
#             count=5000,
#             height=100,
#             width=100,
#             max_object_scale=0.75,
#             transform=image_transform,
#             target_transform=target_transform,
#             **dataset_dict)
#         return train_set

class VATLoss(nn.Module):

    def __init__(self, xi=0.01, eps=1.0, prop_eps=0.25, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.prop_eps = prop_eps

    def forward(self, model: Model, x: torch.Tensor):
        with torch.no_grad():
            pred = model(x)
        assert simplex(pred[0]), f"pred should be simplex."

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                # here the pred_hat is the list of simplex
                adv_distance = list(map(lambda x, y: _kl_div(x, y), pred_hat, pred))
                # adv_distance = _kl_div(F.softmax(pred_hat, dim=1), pred)
                _adv_distance = sum(adv_distance) / float(len(adv_distance))  # type: ignore
                _adv_distance.backward()  # type: ignore
                d = _l2_normalize(d.grad.clone())
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps.view(-1, 1) * self.prop_eps if isinstance(self.eps,
                                                                           torch.Tensor) else d * self.eps * self.prop_eps
            pred_hat = model(x + r_adv)
            lds = list(map(lambda x, y: _kl_div(x, y), pred_hat, pred))
            lds = sum(lds) / float(len(lds))

        return lds, (x + r_adv).detach(), r_adv.detach()


class Trainer(_Trainer):
    def __init__(self, model: Model, train_loader: DataLoader, val_loader: DataLoader, max_epoch: int = 100,
                 save_dir: str = 'MI', use_vat: bool = False, checkpoint_path: str = None, device='cpu',
                 config: dict = None) -> None:
        super().__init__(model, train_loader, val_loader, max_epoch, save_dir, checkpoint_path, device, config)
        self.use_vat = use_vat
        self.criterion = MultualInformaton_IMSAT(mu=4, separate_return=True)
        self.jsd = JSD()
        self.kl = KL_div()
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
        for _batch_num, images_labels_indices in enumerate(_train_loader):
            images, labels, indices = zip(*images_labels_indices)
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

            if not self.use_vat:
                # use transformation
                _sat_loss = list(map(lambda p1, p2: self.kl(p2, p1.detach()), pred_tf1_simplex, pred_tf2_simplex))
                sat_loss = sum(_sat_loss) / len(_sat_loss)
            else:
                sat_loss, *_ = VATLoss(xi=1, eps=10, prop_eps=0.1)(self.model.torchnet, tf1_images)

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
        for _batch_num, images_labels_indices in enumerate(_val_loader):
            images, labels, _ = zip(*images_labels_indices)
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
        # self.writer.add_histogram(tag='probas', values=probas[0], global_step=epoch)

        plt.clf()
        probas = pd.DataFrame(probas.squeeze().cpu().numpy())
        for k in probas.keys():
            probas[k].plot.density(label=str(k))
        plt.legend()
        plt.grid()
        plt.show()
        plt.savefig(self.save_dir / 'distribution.png')
        plt.close('all')
        print(pd.Series(preds.cpu().numpy().ravel()).value_counts())

        return self.METERINTERFACE.val_best_acc.summary()['mean']


config = ConfigManger(DEFAULT_CONFIG_PATH='./config.yml', verbose=True).config

datainterface = MNISTDatasetInterface(**config['DataLoader'])
train_loader = datainterface.ParallelDataLoader(
    default_mnist_img_transform['tf1'],
    default_mnist_img_transform['tf2'],
    default_mnist_img_transform['tf2'],
    default_mnist_img_transform['tf2'],

)
val_loader = datainterface.ParallelDataLoader(
    default_mnist_img_transform['tf3'],
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
