__all__ = ["Model"]
import warnings
from abc import ABC
from typing import *

import torch
from torch import Tensor
from torch import nn
from torch.optim import lr_scheduler

from deepclustering import ModelMode
from deepclustering.arch import get_arch, PlaceholderNet
from .. import optim


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


class Model(ABC):
    def __init__(
            self,
            arch_dict: Dict[str, Any] = None,
            optim_dict: Dict[str, Any] = None,
            scheduler_dict: Dict[str, Any] = None,
    ) -> None:
        super().__init__()
        self.arch_dict = arch_dict
        self.optim_dict = optim_dict
        self.scheduler_dict = scheduler_dict
        self.torchnet, self.optimizer, self.scheduler = self._setup()
        self.to(device=torch.device("cpu"))
        self.is_apex: bool = False

    def _setup(self) -> Tuple[nn.Module, optim.SGD, torch.optim.lr_scheduler.LambdaLR]:
        """
        Initialize torchnet, optimizer, and scheduler based on parameters.
        :return: torchnet, optimizer and scheduler.
        """

        torchnet: nn.Module
        if self.arch_dict is not None:
            self.arch_name = self.arch_dict["name"]
            self.arch_params = {k: v for k, v in self.arch_dict.items() if k != "name"}
            torchnet = get_arch(self.arch_name, self.arch_params)
        else:
            self.arch_name = None
            self.arch_params = None
            warnings.warn(f"torchnet is a placeholder, to override.", RuntimeWarning)
            torchnet = PlaceholderNet()

        # todo: add DataParallel here
        #  this put the tensor to cuda directly, including the forward image implicitly.
        # torchnet = nn.DataParallel(torchnet)

        optimizer: optim.Optimizer
        if self.optim_dict is not None:
            self.optim_name = self.optim_dict["name"]
            self.optim_params = {k: v for k, v in self.optim_dict.items() if k != "name"}
            optimizer = getattr(optim, self.optim_name)(torchnet.parameters(), **self.optim_params)
        else:
            warnings.warn(f"optimizer is a placeholder (lr=0.0), to override.", RuntimeWarning)
            self.optim_name = None
            self.optim_params = None
            optimizer = getattr(optim, "SGD")(torchnet.parameters(), lr=0.00)

        scheduler: lr_scheduler._LRScheduler
        if self.scheduler_dict:
            self.scheduler_name = self.scheduler_dict["name"]
            self.scheduler_params = {k: v for k, v in self.scheduler_dict.items() if k != "name"}
            scheduler = getattr(lr_scheduler, self.scheduler_name) \
                (optimizer, **self.scheduler_params)
        else:
            warnings.warn(f"scheduler is a constant placeholder, to override.", RuntimeWarning)
            self.scheduler_name = None
            self.scheduler_params = None
            scheduler = getattr(lr_scheduler, "StepLR")(optimizer, 1, 1)
        return torchnet, optimizer, scheduler

    @property
    def training(self):
        return self.torchnet.training

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def schedulerStep(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def state_dict(self):
        return {
            "arch_dict": self.arch_dict,
            "optim_dict": self.optim_dict,
            "scheduler_dict": self.scheduler_dict,
            "net_state_dict": self.torchnet.state_dict(),
            "optim_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict: dict):
        self.torchnet.load_state_dict(state_dict["net_state_dict"])
        self.optimizer.load_state_dict(state_dict["optim_state_dict"])
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])

    def to(self, device: torch.device):
        self.torchnet.to(device)
        for state in self.optimizer.state.values():  # type: ignore
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def set_mode(self, mode):
        assert mode in (ModelMode.TRAIN, ModelMode.EVAL) or mode in ("train", "eval")
        if mode in (ModelMode.TRAIN, "train"):
            self.train()
        elif mode in (ModelMode.EVAL, "eval"):
            self.eval()

    def eval(self):
        self.torchnet.eval()

    def train(self):
        self.torchnet.train()

    def __call__(self, *args, **kwargs):
        return self.torchnet(*args, **kwargs)

    def apply(self, *args, **kwargs):
        self.torchnet.apply(*args, **kwargs)

    def __repr__(self):
        return (
            f""
            f"================== Model =================\n"
            f"{self.torchnet.__repr__()}\n"
            f"================== Optimizer =============\n"
            f"{self.optimizer.__repr__()}\n"
            f"================== Scheduler =============\n"
            f"{self.scheduler.__repr__()}\n"
            f"================== Model End ============="
        )

    @classmethod
    def initialize_from_state_dict(cls, state_dict: Dict[str, dict]):
        """
        Initialize an instance based on `state_dict`
        :param state_dict:
        :return: instance model on cpu.
        """

        arch_dict = state_dict["arch_dict"]
        optim_dict = state_dict["optim_dict"]
        scheduler_dict = state_dict["scheduler_dict"]
        model = cls(arch_dict=arch_dict, optim_dict=optim_dict, scheduler_dict=scheduler_dict)
        model.load_state_dict(state_dict=state_dict)
        model.to(torch.device("cpu"))
        return model
