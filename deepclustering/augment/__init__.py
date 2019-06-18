"""
Interface to build the data transformation, taking a dict to
return a corresponding cascaded transformation.
"""
__all__ = ['SequentialWrapper', 'transforms', 'TransformInterface', '_register_transform']
from functools import partial
from typing import *
from .sychronized_augment import SequentialWrapper
from torchvision import transforms
from . import augment
from ..utils.general import _register

TRANSFORM_CALLABLE: Dict[str, Callable] = {}

_register_transform = partial(_register, CALLABLE_DICT=TRANSFORM_CALLABLE)

_register_transform('img2tensor', augment.Img2Tensor)
_register_transform('pilcutout', augment.PILCutout)
_register_transform('randomcrop', augment.RandomCrop)
_register_transform('resize', augment.Resize)
_register_transform('centercrop', augment.CenterCrop)
_register_transform('sobelprocess', augment.SobelProcess)
_register_transform('tolabel', augment.ToLabel)
_register_transform('totensor', augment.ToTensor)

config = {
    'randomcrop': {'size': (20, 20)},
    'resize': {'size': (32, 32)},
    'Img2Tensor': {'include_rgb': False, 'include_grey': True}
}


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


def TransformInterface(config_dict: dict) -> Callable:
    transforms: Callable = _TransformInterface(config_dict)
    return transforms
