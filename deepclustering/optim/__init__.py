# merge the interface for adabound and torch.optim
from torch.optim import *
from torch.optim.optimizer import Optimizer
from .adabound import AdaBound, AdaBoundW
from .radam import RAdam

# todo: deepcopy of an optimizer can ruin the training
