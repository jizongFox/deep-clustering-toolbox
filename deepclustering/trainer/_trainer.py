import atexit
import os
from abc import abstractmethod
from pathlib import Path
from typing import Union, Dict, Any

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter

from ._hooks import HookMixin
from .. import ModelMode, PROJECT_PATH
from ..decorator import lazy_load_checkpoint
from ..meters import MeterInterface, AverageValueMeter
from ..model import Model
from ..utils import _warnings, set_environment, write_yaml, path2Path
from ..writer import SummaryWriter, DataFrameDrawer


class _Trainer:
    """
    Abstract class for a general trainer, which has _train_loop, _eval_loop,load_state, state_dict, and save_checkpoint
    functions. All other trainers are the subclasses of this class.
    """

    RUN_PATH = str(Path(PROJECT_PATH) / "runs")
    ARCHIVE_PATH = str(Path(PROJECT_PATH) / "archives")
    wholemeter_filename = "wholeMeter.csv"
    checkpoint_identifier = "last.pth"
    _METER_INITIALIZED = False

    @lazy_load_checkpoint
    def __init__(
        self,
        model: Model,
        train_loader: Union[DataLoader, _BaseDataLoaderIter],
        val_loader: DataLoader,
        max_epoch: int = 100,
        save_dir: str = "base",
        checkpoint_path: str = None,
        device="cpu",
        config: dict = None,
        *args,
        **kwargs,
    ) -> None:
        _warnings(*args, **kwargs)
        self._model = model
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._save_dir: Path = Path(self.RUN_PATH) / str(save_dir)
        self._save_dir.mkdir(exist_ok=True, parents=True)
        self._checkpoint = checkpoint_path
        self._max_epoch = int(max_epoch)
        self._best_score: float = -1
        self._start_epoch = 0  # whether 0 or loaded from the checkpoint.
        self._device = torch.device(device)
        # debug flag for `Trainer`
        self._debug = bool(os.environ.get("PYDEBUG") == "1")

        if config:
            self._config = config.copy()
            self._config.pop("Config", None)
            write_yaml(self._config, save_dir=self._save_dir, save_name="config.yaml")
            set_environment(config.get("Environment"))

        self.writer = SummaryWriter(str(self._save_dir))
        # register meters to save results
        self.register_meters()

        # close tensorboard writer automatically.
        atexit.register(self.writer.close)

    @abstractmethod
    def register_meters(self, enable_drawer=True) -> None:
        """
        To be overwrited using `self._meter_interface.register_meter` to add the meter
        :return:
        """
        assert self._METER_INITIALIZED is False
        self._meter_interface = MeterInterface()
        self._METER_INITIALIZED = True
        self._meter_interface.register_new_meter(
            "lr", AverageValueMeter(), group_name="train"
        )
        if enable_drawer:
            self._dataframe_drawer = DataFrameDrawer(
                meterinterface=self._meter_interface,
                save_dir=self._save_dir,
                save_name="DataFrameDrawer.png",
            )

    def to(self, device):
        self._model.to(device=device)

    def _start_training(self):
        for epoch in range(self._start_epoch, self._max_epoch):
            if self._model.get_lr() is not None:
                self._meter_interface["lr"].add(self._model.get_lr()[0])
            self.train_loop(train_loader=self._train_loader, epoch=epoch)
            with torch.no_grad():
                current_score = self.eval_loop(self._val_loader, epoch)
            self._model.schedulerStep()
            # save meters and checkpoints
            self._meter_interface.step()
            if hasattr(self, "_dataframe_drawer"):
                self._dataframe_drawer()
            self.save_checkpoint(self.state_dict(), epoch, current_score)
            self._meter_interface.summary().to_csv(self._save_dir / "wholeMeter.csv")

    def start_training(self):
        return self._start_training()

    @abstractmethod
    def _train_loop(
        self,
        train_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
        epoch: int = 0,
        mode=ModelMode.TRAIN,
        *args,
        **kwargs,
    ):
        # warning control
        _warnings(args, kwargs)

    def train_loop(self, *args, **kwargs):
        return self._train_loop(*args, **kwargs)

    @abstractmethod
    def _run_step(self, *args, **kwargs):
        pass

    def run_step(self, *args, **kwargs):
        return self._run_step(*args, **kwargs)

    @abstractmethod
    def _eval_loop(
        self,
        val_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
        epoch: int = 0,
        mode=ModelMode.EVAL,
        *args,
        **kwargs,
    ) -> float:
        # warning control
        _warnings(*args, **kwargs)

    def eval_loop(self, *args, **kwargs):
        return self._eval_loop(*args, **kwargs)

    def inference(self, identifier="best.pth", *args, **kwargs):
        """
        Inference using the checkpoint, to be override by subclasses.
        :param args:
        :param kwargs:
        :return:
        """
        if self._checkpoint is None:
            self._checkpoint = self._save_dir
        assert Path(self._checkpoint).exists(), Path(self._checkpoint)
        assert (Path(self._checkpoint).is_dir() and identifier is not None) or (
            Path(self._checkpoint).is_file() and identifier is None
        )

        state_dict = torch.load(
            str(Path(self._checkpoint) / identifier)
            if identifier is not None
            else self._checkpoint,
            map_location=torch.device("cpu"),
        )
        self.load_checkpoint(state_dict)
        self._model.to(self._device)
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

    def save_checkpoint(
        self, state_dict, current_epoch, cur_score, save_dir=None, save_name=None
    ):
        """
        save checkpoint with adding 'epoch' and 'best_score' attributes
        :param state_dict:
        :param current_epoch:
        :param cur_score:
        :return:
        """
        save_best: bool = True if float(cur_score) > float(self._best_score) else False
        if save_best:
            self._best_score = float(cur_score)
        state_dict["epoch"] = current_epoch
        state_dict["best_score"] = float(self._best_score)
        save_dir = self._save_dir if save_dir is None else path2Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if save_name is None:
            # regular saving
            torch.save(state_dict, str(save_dir / "last.pth"))
            if save_best:
                torch.save(state_dict, str(save_dir / "best.pth"))
        else:
            # periodic saving
            torch.save(state_dict, str(save_dir / save_name))

    def _load_state_dict(self, state_dict) -> None:
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
        self._load_state_dict(state_dict)
        self._best_score = state_dict["best_score"]
        self._start_epoch = state_dict["epoch"] + 1

    def load_checkpoint_from_path(self, checkpoint_path):
        checkpoint_path = path2Path(checkpoint_path)
        assert checkpoint_path.exists(), checkpoint_path
        if checkpoint_path.is_dir():
            state_dict = torch.load(
                str(Path(checkpoint_path) / self.checkpoint_identifier),
                map_location=torch.device("cpu"),
            )
        else:
            assert checkpoint_path.suffix == ".pth", checkpoint_path
            state_dict = torch.load(
                str(checkpoint_path), map_location=torch.device("cpu"),
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
        sub_dir = self._save_dir.relative_to(Path(self.RUN_PATH))
        save_dir = Path(self.ARCHIVE_PATH) / str(sub_dir)
        if Path(save_dir).exists():
            shutil.rmtree(save_dir, ignore_errors=True)
        shutil.move(str(self._save_dir), str(save_dir))
        shutil.rmtree(str(self._save_dir), ignore_errors=True)


class _TrainerHook(HookMixin, _Trainer):
    pass
