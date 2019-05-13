import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepclustering.model import Model
from .Trainer import _Trainer
from .. import ModelMode
from ..loss.IMSAT_loss import Perturbation_Loss, MultualInformaton_IMSAT
from ..meters import AverageValueMeter, MeterInterface
from ..utils import tqdm_, simplex
from ..utils.classification.assignment_mapping import hungarian_match, flat_acc
from ..writer import SummaryWriter, DrawCSV


class IMSATTrainer(_Trainer):
    """
    Trainer specific for IMSAT paper
    """
    METER_CONFIG = {
        'train_sat_loss': AverageValueMeter(),
        'train_mi_loss': AverageValueMeter(),
        'val_acc': AverageValueMeter()
    }
    METERINTERFACE = MeterInterface(METER_CONFIG)

    def __init__(self, model: Model, train_loader: DataLoader, val_loader: DataLoader, max_epoch: int = 1,
                 save_dir: str = './runs/IMSAT', checkpoint_path: str = None, device='cpu',
                 config: dict = None) -> None:
        super().__init__(model, train_loader, val_loader, max_epoch, save_dir, checkpoint_path, device, config)
        self.SAT_criterion = Perturbation_Loss()
        self.MI_criterion = MultualInformaton_IMSAT()

        self.writer = SummaryWriter(str(self.save_dir))
        self.drawer = DrawCSV(columns_to_draw=['train_sat_loss_mean', 'train_mi_loss_mean', 'val_acc_mean'],
                              save_dir=str(self.save_dir), save_name='plot.png')

    def _train_loop(self, train_loader, epoch, mode: ModelMode = ModelMode.TRAIN, **kwargs):
        self.model.set_mode(mode)
        assert self.model.training, f"Model should be in train() model, given {self.model.training}."
        train_loader_: tqdm = tqdm_(train_loader)  # reinitilize the train_loader
        train_loader_.set_description(f'Training epoch: {epoch}')
        for batch, image_labels in enumerate(train_loader_):
            images, _ = list(zip(*image_labels))
            # print(f"used time for dataloading:{time.time() - time_before}")
            tf1_images = torch.cat([images[0] for _ in range(images.__len__() - 1)], dim=0).to(self.device)
            tf2_images = torch.cat(images[1:], dim=0).to(self.device)
            assert tf1_images.shape == tf2_images.shape
            self.model.zero_grad()
            tf1_pred_logit = self.model.torchnet(tf1_images.view(tf1_images.size(0), -1))
            tf2_pred_logit = self.model.torchnet(tf2_images.view(tf2_images.size(0), -1))
            assert not simplex(tf1_pred_logit) and tf1_pred_logit.shape == tf2_pred_logit.shape
            sat_loss = self.SAT_criterion(tf1_pred_logit, tf2_pred_logit)
            ml_loss = self.MI_criterion(tf1_pred_logit)
            # sat_loss = torch.Tensor([0]).cuda()
            batch_loss: torch.Tensor = sat_loss - 0.1 * ml_loss
            self.METERINTERFACE['train_sat_loss'].add(sat_loss.item())
            self.METERINTERFACE['train_mi_loss'].add(ml_loss.item())
            self.model.zero_grad()
            batch_loss.backward()
            self.model.step()
            report_dict = {'sat': self.METERINTERFACE['train_sat_loss'].summary()['mean'],
                           'mi': self.METERINTERFACE['train_mi_loss'].summary()['mean']}
            train_loader_.set_postfix(report_dict)

    def _eval_loop(self, val_loader: DataLoader, epoch: int, mode: ModelMode = ModelMode.EVAL, **kwargs) -> float:
        self.model.set_mode(mode)
        assert not self.model.training, f"Model should be in eval model in _eval_loop, given {self.model.training}."
        val_loader_: tqdm = tqdm_(val_loader)
        preds = torch.zeros(val_loader.dataset.__len__(),
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
            _pred = F.softmax(self.model.torchnet(images.view(images.size(0), -1)), 1)
            assert simplex(_pred)
            bSlicer = slice(slice_done, slice_done + images.shape[0])
            preds[bSlicer] = _pred.max(1)[1]
            target[bSlicer] = gt
            slice_done += gt.shape[0]
        assert slice_done == val_loader.dataset.__len__(), 'Slice not completed.'
        reorder_pred, remap = hungarian_match(
            flat_preds=preds,
            flat_targets=target,
            preds_k=10,
            targets_k=10
        )
        _acc = flat_acc(reorder_pred, target)
        subhead_accs.append(_acc)
        # record average acc
        self.METERINTERFACE.val_acc.add(_acc)

        # record best acc
        report_dict = {'val_acc': self.METERINTERFACE.val_acc.summary()['mean']}
        report_dict_str = ', '.join([f'{k}:{v:.3f}' for k, v in report_dict.items()])
        print(f"Validating epoch: {epoch} : {report_dict_str}")
        return self.METERINTERFACE.val_acc.summary()['mean']

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

    @property
    def state_dict(self):
        state_dictionary = {}
        state_dictionary['model_state_dict'] = self.model.state_dict
        state_dictionary['meter_state_dict'] = self.METERINTERFACE.state_dict
        return state_dictionary

    def save_checkpoint(self, state_dict, current_epoch, best_score):
        save_best: bool = True if best_score > self.best_score else False
        if save_best:
            self.best_score = best_score
        state_dict['epoch'] = current_epoch
        state_dict['best_score'] = self.best_score

        torch.save(state_dict, str(self.save_dir / 'last.pth'))
        if save_best:
            torch.save(state_dict, str(self.save_dir / 'best.pth'))

    def load_checkpoint(self, state_dict):
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.METERINTERFACE.load_state_dict(state_dict['meter_state_dict'])
        self.best_score = state_dict['best_score']
        self._start_epoch = state_dict['epoch'] + 1
