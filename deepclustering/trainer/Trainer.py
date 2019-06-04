import warnings
from abc import ABC, abstractmethod
from copy import deepcopy as dcopy
from math import isnan
from typing import List

import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader

from .. import ModelMode, PROJECT_PATH
from ..meters import MeterInterface
from ..model import Model
from ..utils import flatten_dict, _warnings, dict_filter
from ..writer import SummaryWriter, DrawCSV


class _Trainer(ABC):
    """
    Abstract class for a general trainer, which has _train_loop, _eval_loop,load_state, state_dict, and save_checkpoint
    functions. All other trainers are the subclasses of this class.
    """
    RUN_PATH = str(Path(PROJECT_PATH) / 'runs')
    ARCHIVE_PATH = str(Path(PROJECT_PATH) / 'archives')
    wholemeter_filename = "wholeMeter.csv"

    def __init__(self, model: Model, train_loader: DataLoader, val_loader: DataLoader, max_epoch: int = 100,
                 save_dir: str = 'base', checkpoint_path: str = None, device='cpu', config: dict = None,
                 **kwargs) -> None:
        super().__init__()
        _warnings((), kwargs)
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir: Path = Path(self.RUN_PATH) / save_dir
        self.save_dir.mkdir(exist_ok=True, parents=True)
        (self.save_dir / 'meters').mkdir(exist_ok=True, parents=True)
        self.checkpoint = checkpoint_path
        self.max_epoch = int(max_epoch)
        self.best_score: float = -1
        self._start_epoch = 0  # whether 0 or loaded from the checkpoint.
        self.device = torch.device(device)

        if config:
            self.config = dcopy(config)
            self.config.pop('Config', None)  # delete the Config attribute
            with open(self.save_dir / 'config.yaml', 'w') as outfile:
                yaml.dump(self.config, outfile, default_flow_style=False)

        self.writer = SummaryWriter(str(self.save_dir))
        # todo: try to override the DrawCSV
        _columns_to_draw = self.__init_meters__()
        self.drawer = DrawCSV(columns_to_draw=_columns_to_draw,
                              save_dir=str(self.save_dir),
                              save_name='plot.png',
                              csv_name=self.wholemeter_filename
                              )
        if checkpoint_path:
            assert Path(checkpoint_path).exists() and Path(checkpoint_path).is_dir(), Path(checkpoint_path)
            state_dict = torch.load(str(Path(checkpoint_path) / 'last.pth'), map_location=torch.device('cpu'))
            self.load_checkpoint(state_dict)
        self.model.to(self.device)

    @abstractmethod
    def __init_meters__(self) -> List[str]:
        METER_CONFIG = {}
        self.METERINTERFACE = MeterInterface(METER_CONFIG)
        return ["draw_columns_list"]

    @property
    @abstractmethod
    def _training_report_dict(self):
        report_dict = flatten_dict({})
        report_dict = dict_filter(report_dict, lambda k, v: 1 - isnan(v))
        return report_dict

    @property
    @abstractmethod
    def _eval_report_dict(self):
        report_dict = flatten_dict({})
        report_dict = dict_filter(report_dict, lambda k, v: 1 - isnan(v))
        return report_dict

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
            self.METERINTERFACE.summary().to_csv(self.save_dir / self.wholemeter_filename)
            self.writer.add_scalars('Scalars', self.METERINTERFACE.summary().iloc[-1].to_dict(), global_step=epoch)
            self.drawer.call_draw()
            self.save_checkpoint(self.state_dict, epoch, current_score)

    def to(self, device):
        self.model.to(device=device)

    @abstractmethod
    def _train_loop(self, train_loader=None, epoch: int = 0, mode=ModelMode.TRAIN, *args, **kwargs):
        # warning control
        _warnings(args, kwargs)

    def _trainer_specific_loss(self, *args, **kwargs):
        # warning control
        _warnings(args, kwargs)

    @abstractmethod
    def _eval_loop(self, val_loader: DataLoader = None, epoch: int = 0, mode=ModelMode.EVAL, *args, **kwargs) -> float:
        # warning control
        _warnings(args, kwargs)

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
        try:
            self.METERINTERFACE.load_state_dict(state_dict['meter_state_dict'])
        except KeyError:
            warnings.warn('Meter checkpoint does not match.')

        self.best_score = state_dict['best_score']
        self._start_epoch = state_dict['epoch'] + 1

    def clean_up(self):
        import shutil
        import time
        time.sleep(10)  # to prevent that the call_draw function is not ended.
        Path(self.ARCHIVE_PATH).mkdir(exist_ok=True, parents=True)
        sub_dir = self.save_dir.relative_to(Path(self.RUN_PATH))
        save_dir = Path(self.ARCHIVE_PATH) / str(sub_dir)
        if Path(save_dir).exists():
            shutil.rmtree(save_dir, ignore_errors=True)
        shutil.move(str(self.save_dir), str(save_dir))
        shutil.rmtree(str(self.save_dir), ignore_errors=True)
