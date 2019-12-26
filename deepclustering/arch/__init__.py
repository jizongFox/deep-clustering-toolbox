__all__ = [
    "weights_init",
    "get_arch",
    "ARCH_CALLABLES",
    "PlaceholderNet",
    "_register_arch",
]

from functools import partial
from typing import *

from .classification import *
from .segmentation import *
from ..utils.general import _register

"""
Package
"""
# A Map from string to arch callables
ARCH_CALLABLES: Dict[str, Callable] = {}
ARCH_PARAM_DICT: Dict[str, Dict[str, Union[int, float, str]]] = {}

_register_arch = partial(_register, CALLABLE_DICT=ARCH_CALLABLES)
_register_param = partial(_register_arch, CALLABLE_DICT=ARCH_PARAM_DICT)

# Adding architecture (new architecture goes here...)
_register_arch("clusternet5g", ClusterNet5g)
_register_arch("clusternet5gtwohead", ClusterNet5gTwoHead)
_register_arch("clusternet5gmultihead", ClusterNet5gMultiHead)
_register_arch("clusternet6c", ClusterNet6c)
_register_arch("clusternet6cTwoHead", ClusterNet6cTwoHead)
_register_arch("clusternetimsat", IMSATNet)
_register_arch("dummy", Dummy)
_register_arch("vatnet", VATNetwork)
_register_arch("enet", Enet)
_register_arch("unet", UNet)
_register_arch("unet_bn", UNet_bn)
_register_arch("cnet", CorstemNet)
_register_arch("preresnet", PreResNet)
_register_arch("epsnetv2", ESPNetv2)
# Adding default keys here to enable automatic testing
_register_param("clusternet5g", ClusterNet5g_Param)
_register_param("clusternet5gtwohead", ClusterNet5gTwoHead_Param)
_register_param("clusternet5gmultihead", ClusterNet5gMultiHead_Param)
_register_param("clusternet6c", ClusterNet6c_Param)
_register_param("clusternet6cTwoHead", ClusterNet6cTwoHead_Param)
_register_param("clusternetimsat", IMSATNet_Param)
_register_param("dummy", Dummy_Param)
_register_param("vatnet", VatNet_Param)
_register_param("enet", Enet_Param)
_register_param("unet", Unet_Param)
_register_param("unet_bn", Unetbn_Param)
_register_param("cnet", CorstemNet_Param)
_register_param("preresnet", PreResNet110_params)
"""
Public interface
"""


def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal_(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


def get_arch(arch: str, kwargs) -> nn.Module:
    """ Get the architecture. Return a torch.nn.Module """
    arch_callable = ARCH_CALLABLES.get(arch.lower())
    kwargs.pop("arch", None)
    assert arch_callable, "Architecture {} is not found!".format(arch)
    net = arch_callable(**kwargs)
    # try:
    #     net.apply(weights_init)
    # except AttributeError as e:
    #     print(f'Using pretrained models with the error:{e}')
    return net
