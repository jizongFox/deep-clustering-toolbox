import warnings
from abc import ABC
from typing import *

import torch
from torch import Tensor
from torch import nn
from torch.nn import NLLLoss
from torch.nn import functional as F
from torch.optim import lr_scheduler

from deepclustering import ModelMode
from deepclustering.arch import get_arch, PlaceholderNet
# from torch import optim
from .. import optim


class Model(ABC):

    def __init__(
            self,
            arch_dict: Dict[str, Any] = None,
            optim_dict: Dict[str, Any] = None,
            scheduler_dict: Dict[str, Any] = None
    ) -> None:
        super().__init__()
        self.arch_dict = arch_dict
        self.optim_dict = optim_dict
        self.scheduler_dict = scheduler_dict
        self.torchnet, self.optimizer, self.scheduler = self._setup()
        self.to(device=torch.device('cpu'))

    def _setup(self) -> Tuple[nn.Module, optim.SGD, torch.optim.lr_scheduler.LambdaLR]:
        if self.arch_dict is not None:
            self.arch_name = self.arch_dict['name']
            self.arch_params = {k: v for k, v in self.arch_dict.items() if k != 'name'}
            torchnet = get_arch(self.arch_name, self.arch_params)
        else:
            warnings.warn(f'torchnet is a placeholder, override it later.', RuntimeWarning)
            self.arch_name = None
            self.arch_params = None
            torchnet = PlaceholderNet()
        # this put the tensor to cuda directly, including the forward image implicitly.
        # torchnet = nn.DataParallel(torchnet)
        if self.optim_dict is not None:
            self.optim_name = self.optim_dict['name']
            self.optim_params = {k: v for k, v in self.optim_dict.items() if k != 'name'}
            optimizer: optim.SGD = getattr(optim, self.optim_name) \
                (torchnet.parameters(), **self.optim_params)
        else:
            warnings.warn(f'optimizer is a placeholder, override it later.', RuntimeWarning)
            self.optim_name = None
            self.optim_params = None
            optimizer: optim.SGD = getattr(optim, 'SGD')(torchnet.parameters(), lr=0.01)

        if self.scheduler_dict is not None:
            self.scheduler_name = self.scheduler_dict['name']
            self.scheduler_params = {k: v for k, v in self.scheduler_dict.items() if
                                     k != 'name'}
            scheduler: lr_scheduler.LambdaLR = getattr(lr_scheduler, self.scheduler_name) \
                (optimizer, **self.scheduler_params)
        else:
            warnings.warn(f'scheduler is a placeholder, override it later.', RuntimeWarning)
            self.scheduler_name = None
            self.scheduler_params = None
            scheduler: lr_scheduler.LambdaLR = getattr(lr_scheduler, 'StepLR') \
                (optimizer, 10, 0.1)

        return torchnet, optimizer, scheduler

    def predict(self, img: Tensor, logit=True) -> Tensor:
        pred_logit = self.torchnet(img)
        if logit:
            return pred_logit
        return F.softmax(pred_logit, 1)

    @property
    def training(self):
        return self.torchnet.training

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def update(self, img: Tensor, gt: Tensor, criterion: NLLLoss,
               mode=ModelMode.TRAIN) -> List[Tensor]:
        # todo improve the code
        assert img.shape.__len__() == 4
        assert gt.shape.__len__() == 4
        if mode == ModelMode.TRAIN:
            self.train()
        else:
            self.eval()

        if mode == ModelMode.TRAIN:
            self.optimizer.zero_grad()
            pred = self.predict(img)
            loss = criterion(pred, gt.squeeze(1))
            loss.backward()
            self.optimizer.step()
        else:
            with torch.no_grad():
                pred = self.predict(img)
                loss = criterion(pred, gt.squeeze(1))
        self.train()
        return [pred.detach(), loss.detach()]

    def schedulerStep(self):
        if self.scheduler is not None:
            self.scheduler.step()

    @property
    def state_dict(self):
        return {
            'arch_dict': self.arch_dict,
            'optim_dict': self.optim_dict,
            'scheduler_dict': self.scheduler_dict,
            'net_state_dict': self.torchnet.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }

    def load_state_dict(self, state_dict: dict):
        self.torchnet.load_state_dict(state_dict['net_state_dict'])
        self.optimizer.load_state_dict(state_dict['optim_state_dict'])
        self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])

    def to(self, device: torch.device):
        self.torchnet.to(device)
        for state in self.optimizer.state.values():  # type: ignore
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def set_mode(self, mode):
        assert mode in (ModelMode.TRAIN, ModelMode.EVAL) or mode in ('train', 'eval')
        if mode in (ModelMode.TRAIN, 'train'):
            self.train()
        elif mode in (ModelMode.EVAL, 'eval'):
            self.eval()

    def eval(self):
        self.torchnet.eval()

    def train(self):
        self.torchnet.train()

    def __call__(self, img: Tensor, logit=True):
        return self.predict(img=img, logit=logit)

    @classmethod
    def initialize_from_state_dict(cls, state_dict: Dict[str, dict]):
        arch_dict = state_dict['arch_dict']
        optim_dict = state_dict['optim_dict']
        scheduler_dict = state_dict['scheduler_dict']
        model = cls(arch_dict=arch_dict, optim_dict=optim_dict, scheduler_dict=scheduler_dict)
        model.load_state_dict(state_dict=state_dict)
        model.to(torch.device('cpu'))
        return model

    def apply(self, *args, **kwargs):
        self.torchnet.apply(*args, **kwargs)

    def __repr__(self):
        return f"" \
            f"================== Model =================\n" \
            f"{self.torchnet.__repr__()}\n" \
            f"================== Optimizer =============\n" \
            f"{self.optimizer.__repr__()}\n" \
            f"================== Scheduler =============\n" \
            f"{self.scheduler.__repr__()}\n" \
            f"================== Model End ============="
