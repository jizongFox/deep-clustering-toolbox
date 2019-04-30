from abc import ABC
from typing import *

import torch
from generalframework import ModelMode
from generalframework.arch import get_arch
from torch import Tensor
from torch import nn
from torch import optim
from torch.nn import NLLLoss
from torch.nn import functional as F
from torch.optim import lr_scheduler


class Model(ABC):

    def __init__(self, arch_dict: Dict[str, Any], optim_dict: Dict[str, Any], scheduler_dict: Dict[str, Any]) -> None:
        super().__init__()
        self.arch_dict = arch_dict
        self.optim_dict = optim_dict
        self.scheduler_dict = scheduler_dict
        self.torchnet, self.optimizer, self.scheduler = self.__setup()

    def __setup(self) -> Tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        self.arch_name = self.arch_dict['name']
        self.arch_params = {k: v for k, v in self.arch_dict.items() if k != 'name'}
        self.optim_name = self.optim_dict['name']
        self.optim_params = {k: v for k, v in self.optim_dict.items() if k != 'name'}
        self.scheduler_name = self.scheduler_dict['name']
        self.scheduler_params = {k: v for k, v in self.scheduler_dict.items() if k != 'name'}
        torchnet = get_arch(self.arch_name, self.arch_params)
        torchnet = nn.DataParallel(torchnet)
        optimizer: optim.Optimizer = getattr(optim, self.optim_name) \
            (torchnet.parameters(), **self.optim_params)
        scheduler: lr_scheduler.LambdaLR = getattr(lr_scheduler, self.scheduler_name) \
            (optimizer, **self.scheduler_params)
        return torchnet, optimizer, scheduler

    def predict(self, img: Tensor, logit=True) -> Tensor:
        pred_logit = self.torchnet(img)
        if logit:
            return pred_logit
        return F.softmax(pred_logit, 1)

    @property
    def training(self):
        return self.torchnet.training

    def update(self, img: Tensor, gt: Tensor, criterion: NLLLoss,
               mode=ModelMode.TRAIN) -> List[Tensor]:
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
        return [pred.data, loss.data]

    def schedulerStep(self):
        self.scheduler.step()

    @property
    def state_dict(self):
        return {'arch_dict': self.arch_dict, 'optim_dict': self.optim_dict, 'scheduler_dict': self.scheduler_dict,
                'net_state_dict': self.torchnet.state_dict(), 'optim_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()}

    def load_state_dict(self, state_dict: dict):
        try:
            self.torchnet.load_state_dict(state_dict['net_state_dict'])
        except:
            self.torchnet.load_state_dict(
                {k.replace("module.", ""): v for k, v in state_dict['net_state_dict'].items()}
            )
        self.optimizer.load_state_dict(state_dict['optim_state_dict'])
        self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])

    def to(self, device: torch.device):
        self.torchnet.to(device)

        for state in self.optimizer.state.values():
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
