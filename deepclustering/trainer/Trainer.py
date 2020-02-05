import atexit
import os
from abc import ABC, abstractmethod
from copy import deepcopy as dcopy
from pathlib import Path
from typing import List, Union, Dict, Any

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

from .. import ModelMode, PROJECT_PATH
from ..decorator import lazy_load_checkpoint
from ..meters import MeterInterface
from ..model import Model
from ..utils import flatten_dict, _warnings, dict_filter, set_environment
from ..writer import SummaryWriter, DrawCSV2


class _Trainer(ABC):
    """
    Abstract class for a general trainer, which has _train_loop, _eval_loop,load_state, state_dict, and save_checkpoint
    functions. All other trainers are the subclasses of this class.
    """

    RUN_PATH = str(Path(PROJECT_PATH) / "runs")
    ARCHIVE_PATH = str(Path(PROJECT_PATH) / "archives")
    wholemeter_filename = "wholeMeter.csv"
    checkpoint_identifier = "last.pth"

    @lazy_load_checkpoint
    def __init__(
        self,
        model: Model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epoch: int = 100,
        save_dir: str = "base",
        checkpoint_path: str = None,
        device="cpu",
        config: dict = None,
        **kwargs,
    ) -> None:
        _warnings((), kwargs)
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir: Path = Path(self.RUN_PATH) / save_dir
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint = checkpoint_path
        self.max_epoch = int(max_epoch)
        self.best_score: float = -1
        self._start_epoch = 0  # whether 0 or loaded from the checkpoint.
        self.device = torch.device(device)
        # debug flag for `Trainer`
        self._debug = bool(os.environ.get("PYDEBUG") == "1")

        if config:
            self.config = dcopy(config)
            self.config.pop("Config", None)  # delete the Config attribute
            with open(
                str(self.save_dir / "config.yaml"), "w"
            ) as outfile:  # type: ignore
                yaml.dump(self.config, outfile, default_flow_style=False)
            # set environment variable:
            set_environment(config.get("Environment"))

        self.writer = SummaryWriter(str(self.save_dir))
        # todo: try to override the DrawCSV
        _columns_to_draw = self.__init_meters__()
        self.drawer = DrawCSV2(
            columns_to_draw=_columns_to_draw,
            save_dir=str(self.save_dir),
            save_name="plot.png",
            csv_name=self.wholemeter_filename,
        )
        atexit.register(self.writer.close)

    @abstractmethod
    def __init_meters__(self) -> List[Union[str, List[str]]]:
        METER_CONFIG = {}
        self.METERINTERFACE = MeterInterface(METER_CONFIG)
        return ["draw_columns_list"]

    @property
    @abstractmethod
    def _training_report_dict(self):
        report_dict = flatten_dict({})
        report_dict = dict_filter(report_dict)
        return report_dict

    @property
    @abstractmethod
    def _eval_report_dict(self):
        report_dict = flatten_dict({})
        report_dict = dict_filter(report_dict)
        return report_dict

    def start_training(self):
        for epoch in range(self._start_epoch, self.max_epoch):
            self._train_loop(train_loader=self.train_loader, epoch=epoch)
            with torch.no_grad():
                current_score = self._eval_loop(self.val_loader, epoch)
            self.METERINTERFACE.step()
            self.model.schedulerStep()
            # save meters and checkpoints
            SUMMARY = self.METERINTERFACE.summary()
            SUMMARY.to_csv(self.save_dir / self.wholemeter_filename)
            self.drawer.draw(SUMMARY)
            self.save_checkpoint(self.state_dict(), epoch, current_score)
        self.writer.close()

    def to(self, device):
        self.model.to(device=device)

    @abstractmethod
    def _train_loop(
        self,
        train_loader: DataLoader = None,
        epoch: int = 0,
        mode=ModelMode.TRAIN,
        *args,
        **kwargs,
    ):
        # warning control
        _warnings(args, kwargs)

    def _trainer_specific_loss(self, *args, **kwargs) -> Tensor:
        # warning control
        _warnings(args, kwargs)

    @abstractmethod
    def _eval_loop(
        self,
        val_loader: DataLoader = None,
        epoch: int = 0,
        mode=ModelMode.EVAL,
        *args,
        **kwargs,
    ) -> float:
        # warning control
        _warnings(args, kwargs)

    def inference(self, *args, **kwargs):
        """
        Inference using the checkpoint, to be override by subclasses.
        :param args:
        :param kwargs:
        :return:
        """
        assert Path(self.checkpoint).exists() and Path(self.checkpoint).is_dir(), Path(
            self.checkpoint
        )
        state_dict = torch.load(
            str(Path(self.checkpoint) / "best.pth"), map_location=torch.device("cpu")
        )
        self.load_checkpoint(state_dict)
        self.model.to(self.device)
        # to be added
        # probably call self._eval() method.

    def state_dict(self) -> Dict[str, Any]:
        """
        return trainer's state dict. The dict is built by considering all the submodules having `state_dict` method.
        """
        state_dictionary = {}
        for module_name, module in self.__dict__.items():
            if hasattr(module, "state_dict"):
                state_dictionary[module_name] = module.state_dict()
        return state_dictionary

    def save_checkpoint(self, state_dict, current_epoch, best_score):
        """
        save checkpoint with adding 'epoch' and 'best_score' attributes
        :param state_dict:
        :param current_epoch:
        :param best_score:
        :return:
        """
        save_best: bool = True if float(best_score) > float(self.best_score) else False
        if save_best:
            self.best_score = float(best_score)
        state_dict["epoch"] = current_epoch
        state_dict["best_score"] = float(self.best_score)

        torch.save(state_dict, str(self.save_dir / "last.pth"))
        if save_best:
            torch.save(state_dict, str(self.save_dir / "best.pth"))

    def load_state_dict(self, state_dict) -> None:
        """
        Load state_dict for submodules having "load_state_dict" method.
        :param state_dict:
        :return:
        """
        for module_name, module in self.__dict__.items():
            if hasattr(module, "load_state_dict"):
                try:
                    module.load_state_dict(state_dict[module_name])
                except KeyError as e:
                    print(f"Loading checkpoint error for {module_name}, {e}.")
                except RuntimeError as e:
                    print(f"Interface changed error for {module_name}, {e}")

    def load_checkpoint(self, state_dict) -> None:
        """
        load checkpoint to models, meters, best score and _start_epoch
        Can be extended by add more state_dict
        :param state_dict:
        :return:
        """
        self.load_state_dict(state_dict)
        self.best_score = state_dict["best_score"]
        self._start_epoch = state_dict["epoch"] + 1

    def load_checkpoint_from_path(self, checkpoint_path):
        assert Path(checkpoint_path).exists() and Path(checkpoint_path).is_dir(), Path(
            checkpoint_path
        )
        state_dict = torch.load(
            str(Path(checkpoint_path) / self.checkpoint_identifier),
            map_location=torch.device("cpu"),
        )
        self.load_checkpoint(state_dict)

    def clean_up(self, wait_time=3):
        """
        Do not touch
        :return:
        """
        import shutil
        import time

        time.sleep(wait_time)  # to prevent that the call_draw function is not ended.
        Path(self.ARCHIVE_PATH).mkdir(exist_ok=True, parents=True)
        sub_dir = self.save_dir.relative_to(Path(self.RUN_PATH))
        save_dir = Path(self.ARCHIVE_PATH) / str(sub_dir)
        if Path(save_dir).exists():
            shutil.rmtree(save_dir, ignore_errors=True)
        shutil.move(str(self.save_dir), str(save_dir))
        shutil.rmtree(str(self.save_dir), ignore_errors=True)
