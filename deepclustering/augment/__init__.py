"""
Interface to build the data transformation, taking a dict to return a corresponding cascaded transformation.
"""
from typing import *

from torchvision import transforms
from ..utils.general import _register
from functools import partial
from . import augment

TRANSFORM_CALLABLE: Dict[str, Callable] = {}

_register_transform = partial(_register, CALLABLE_DICT=TRANSFORM_CALLABLE)

_register_transform('img2tensor', augment.Img2Tensor)
_register_transform('pilcutout', augment.PILCutout)
_register_transform('randomcrop', augment.RandomCrop)
_register_transform('resize', augment.Resize)
_register_transform('centercrop', augment.CenterCrop)
_register_transform('sobelprocess', augment.SobelProcess)

config = {
    'randomcrop': {'size': (20, 20)},
    'resize': {'size': (32, 32)},
    'Img2Tensor': {'include_rgb': False, 'include_grey': True}
}

def TransformInterface(config_dict):
    transforms ={}
    for k, v in config_dict.items():
        transforms[k]=_TransformInterface(v)
    return transforms


def _TransformInterface(config_dict: Dict[str, Any]):
    transformList = []
    for k, v in config_dict.items():
        try:
            t = TRANSFORM_CALLABLE[k.lower()](**v)
        except KeyError:
            t = getattr(transforms, k)(**v)
        transformList.append(t)
    transform = transforms.Compose(transformList)
    return transform
