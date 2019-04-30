from functools import partial
from typing import *

from torch import nn

from .classification import *
from .segmentation import *

__all__ = ['weights_init', 'get_arch', 'ARCH_CALLABLES']
"""
Package
"""
# A Map from string to arch callables
ARCH_CALLABLES: Dict[str, Callable] = {}
ARCH_PARAM_DICT: Dict[str, Dict[str,Union[int, float, str]]] = {}


# meta function
def _register(name: str, callable: Callable, alias=None, CALLABLE_DICT: dict = {}) -> None:
    """ Private method to register the architecture to the ARCH_CALLABLES
        :param name: A str
        :param callable: The callable that return the nn.Module
        :param alias: None, or a list of string, or str
    """
    if name in CALLABLE_DICT:
        raise ValueError('{} already exists!'.format(name.lower()))
    CALLABLE_DICT[name.lower()] = callable
    if alias:
        if isinstance(alias, str):
            alias = [alias]
        for other_arch in alias:
            if other_arch.lower() in CALLABLE_DICT:
                raise ValueError('alias {} for {} already exists!'.format(other_arch.lower(), name.lower()))
            CALLABLE_DICT[other_arch.lower()] = callable


_register_arch = partial(_register, CALLABLE_DICT=ARCH_CALLABLES)
_register_param = partial(_register_arch, CALLABLE_DICT=ARCH_PARAM_DICT)

# Adding architecture (new architecture goes here...)
_register_arch('clusternet5g', ClusterNet5g)
_register_arch('clusternet5gtwohead', ClusterNet5gTwoHead)
_register_arch('clusternet6c', ClusterNet6c)
_register_arch('clusternet6cTwoHead', ClusterNet6cTwoHead)
# Adding default keys here to enable automatic testing
_register_param('clusternet5g', ClusterNet5g_Param)
_register_param('clusternet5gtwohead', ClusterNet5gTwoHead_Param)
_register_param('clusternet6c', ClusterNet6c_Param)
_register_param('clusternet6cTwoHead', ClusterNet6cTwoHead_Param)
"""
Public interface
"""


def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal_(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_arch(arch: str, kwargs) -> nn.Module:
    """ Get the architecture. Return a torch.nn.Module """
    arch_callable = ARCH_CALLABLES.get(arch.lower())
    try:
        kwargs.pop('arch')
    except KeyError:
        pass
    assert arch_callable, "Architecture {} is not found!".format(arch)
    net = arch_callable(**kwargs)
    try:
        net.apply(weights_init)
    except AttributeError as e:
        print(f'Using pretrained models with the error:{e}')
    return net
