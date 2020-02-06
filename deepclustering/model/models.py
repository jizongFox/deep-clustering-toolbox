import warnings
from typing import *

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.optim import lr_scheduler

from deepclustering import ModelMode
from deepclustering import optim
from deepclustering.arch import get_arch
from deepclustering.utils import simplex

__all__ = ["Model", "DeployModel"]


class NormalGradientBackwardStep(object):
    """effectuate the
    model.zero() at the initialization
    and model.step at the exit
    """

    def __init__(self, loss: Tensor, model):
        self.model = model
        self.loss = loss
        self.model.zero_grad()

    def __enter__(self):
        return self.loss

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.step()


CType = Dict[str, Union[float, int, str, Dict[str, Any]]]  # typing for config
NetType = nn.Module
OptimType = optim.Optimizer
ScheType = optim.lr_scheduler._LRScheduler


class Model:
    """
    This is the new class for model interface
    """

    def __init__(
        self,
        arch: Union[NetType, CType],
        optimizer: Union[OptimType, CType] = None,
        scheduler: Union[ScheType, CType] = None,
    ):
        """
        create network from either configuration or module directly.
        :param arch: network configuration or network module
        :param optimizer:
        :param scheduler:
        :return:
        """
        self._set_arch(arch)
        self._set_optimizer(optimizer)
        self._set_scheduler(scheduler)

    def _set_arch(self, arch: Union[NetType, CType]) -> None:
        self._torchnet: nn.Module
        self._arch_dict: Optional[CType]
        if isinstance(arch, dict):
            self._arch_dict = arch.copy()
            _arch_name: str = arch["name"]  # type:ignore
            _arch_params = {k: v for k, v in arch.items() if k != "name"}
            self._torchnet = get_arch(_arch_name, _arch_params)
        else:
            self._arch_dict = None
            self._torchnet = arch
        assert issubclass(type(self._torchnet), nn.Module)

    def _set_optimizer(self, optimizer: Union[OptimType, CType] = None) -> None:
        self._optimizer: Optional[OptimType]
        self._optim_dict: Optional[CType]
        if optimizer is None:
            self._optim_dict = None
            self._optimizer = None
        elif isinstance(optimizer, dict):
            self._optim_dict = optimizer.copy()
            _optim_name: str = optimizer["name"]  # type:ignore
            _optim_params = {k: v for k, v in optimizer.items() if k != "name"}
            self._optimizer = getattr(optim, _optim_name)(
                self.parameters(), **_optim_params
            )
        else:
            self._optim_dict = None
            self._optimizer = optimizer
        if optimizer is not None:
            assert issubclass(type(self._optimizer), optim.Optimizer)

    def _set_scheduler(self, scheduler: Union[ScheType, CType] = None) -> None:
        self._scheduler: Optional[ScheType]
        self._scheduler_dict: Optional[CType]
        if scheduler is None:
            self._scheduler = None
            self._scheduler_dict = None
        elif isinstance(scheduler, dict):
            self._scheduler_dict = scheduler.copy()
            _scheduler_name: str = scheduler["name"]  # type:ignore
            _scheduler_params = {k: v for k, v in scheduler.items() if k != "name"}
            self._scheduler = getattr(lr_scheduler, _scheduler_name)(
                self._optimizer,
                **{k: v for k, v in _scheduler_params.items() if k != "warmup"},
            )
            if "warmup" in _scheduler_params:
                # encode warmup scheduler
                from ..schedulers import GradualWarmupScheduler

                self._scheduler = GradualWarmupScheduler(
                    optimizer=self._optimizer,
                    **_scheduler_params["warmup"],
                    after_scheduler=self._scheduler,
                )
        else:
            self._scheduler_dict = None
            self._scheduler = scheduler
        if scheduler is not None:
            assert issubclass(type(self._scheduler), ScheType)

    def parameters(self):
        return self._torchnet.parameters()

    def __call__(self, *args, **kwargs):
        force_simplex = kwargs.get("force_simplex", False)
        assert isinstance(force_simplex, bool), force_simplex
        torch_logits = self._torchnet(*args, **kwargs)
        if force_simplex:
            if not simplex(torch_logits, 1):
                return F.softmax(torch_logits, 1)
        return torch_logits

    @property
    def training(self):
        return self._torchnet.training

    def step(self):
        if self._optimizer is not None and hasattr(self._optimizer, "step"):
            self._optimizer.step()

    def zero_grad(self) -> None:
        if self._optimizer is not None and hasattr(self._optimizer, "zero_grad"):
            self._optimizer.zero_grad()

    def schedulerStep(self, *args, **kwargs):
        if hasattr(self._scheduler, "step"):
            self._scheduler.step(*args, **kwargs)

    def set_mode(self, mode):
        assert mode in (ModelMode.TRAIN, ModelMode.EVAL) or mode in ("train", "eval")
        if mode in (ModelMode.TRAIN, "train"):
            self.train()
        elif mode in (ModelMode.EVAL, "eval"):
            self.eval()

    def train(self):
        self._torchnet.train()

    def eval(self):
        self._torchnet.eval()

    def to(self, device: torch.device):
        self._torchnet.to(device)
        if self._optimizer is not None:
            for state in self._optimizer.state.values():  # type: ignore
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

    def apply(self, *args, **kwargs) -> None:
        self._torchnet.apply(*args, **kwargs)

    def get_lr(self):
        if self._scheduler is not None:
            return self._scheduler.get_lr()
        return None

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def optimizer_params(self):
        return self._optim_dict

    @property
    def scheduler_params(self):
        return self._scheduler_dict

    def __repr__(self):
        model_descript = (
            f"================== Model =================\n"
            f"{self._torchnet.__repr__()}\n"
        )
        optimizer_descript = (
            f"================== Optimizer =============\n"
            f"{self._optimizer.__repr__()}\n"
            if self._optimizer is not None
            else ""
        )
        scheduler_descript = (
            f"================== Scheduler =============\n"
            f"{self._scheduler.__repr__()}\n"
            if self._scheduler is not None
            else ""
        )

        return model_descript + optimizer_descript + scheduler_descript

    def state_dict(self):
        return {
            "arch_dict": self._arch_dict,
            "optim_dict": self._optim_dict,
            "scheduler_dict": self._scheduler_dict,
            "net_state_dict": self._torchnet.state_dict(),
            "optim_state_dict": self._optimizer.state_dict()
            if self._optimizer is not None
            else None,
            "scheduler_state_dict": self._scheduler.state_dict()
            if self._scheduler is not None
            else None,
        }

    def load_state_dict(self, state_dict: dict):
        self._arch_dict = state_dict["arch_dict"]
        self._optim_dict = state_dict["optim_dict"]
        self._scheduler_dict = state_dict["scheduler_dict"]
        self._torchnet.load_state_dict(state_dict["net_state_dict"])
        if hasattr(self._optimizer, "load_state_dict") and self._optimizer is not None:
            self._optimizer.load_state_dict(state_dict["optim_state_dict"])
        if hasattr(self._scheduler, "load_state_dict") and self._scheduler is not None:
            self._scheduler.load_state_dict(state_dict["scheduler_state_dict"])

    @classmethod
    def initialize_from_state_dict(cls, state_dict: Dict[str, dict]):
        """
        Initialize an instance based on `state_dict`
        :param state_dict:
        :return: instance model on cpu.
        """
        arch_dict = state_dict["arch_dict"]
        assert (
            arch_dict is not None
        ), "arch is only supported when it is initialized with config."
        optim_dict = state_dict["optim_dict"]
        if optim_dict is None:
            warnings.warn(
                f"optim is ignored as it is not initialized with config, use `load_state_dict` instead.",
                RuntimeWarning,
            )
        scheduler_dict = state_dict["scheduler_dict"]
        if scheduler_dict is None:
            warnings.warn(
                f"scheduler is ignored as it is not initialized with config, use `load_state_dict` instead.",
                RuntimeWarning,
            )
        model = cls(arch=arch_dict, optimizer=optim_dict, scheduler=scheduler_dict)
        model.load_state_dict(state_dict=state_dict)
        model.to(torch.device("cpu"))
        return model


class DeployModel(Model):
    def __init__(self, arch: Union[NetType, CType]):
        super().__init__(arch)
        self.eval()

    def step(self):
        raise RuntimeError(f"{self.__class__.__name__} does not support `step` method.")

    def schedulerStep(self, *args, **kwargs):
        raise RuntimeError(
            f"{self.__class__.__name__} does not support `schedulerStep` method."
        )
