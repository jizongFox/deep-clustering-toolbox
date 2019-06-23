import torch

__all__ = ['FixRandomSeed', 'SequentialWrapper']

import random
from typing import Callable, List, Union

import numpy as np
from PIL import Image

from ..utils import identical


class FixRandomSeed(object):

    def __init__(self, random_seed: int = 0):
        self.random_seed = random_seed
        self.randombackup = random.getstate()
        self.npbackup = np.random.get_state()

    def __enter__(self):
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def __exit__(self, *_):
        np.random.set_state(self.npbackup)
        random.setstate(self.randombackup)


class SequentialWrapper(object):
    """
    This is the wrapper for synchronized image transformation
    The idea is to define two transformations for images and targets, with randomness.
    The randomness is garanted by the same random seed
    """

    def __init__(self, img_transform: Callable = None, target_transform: Callable = None,
                 if_is_target: List[bool] = []) -> None:
        super().__init__()
        self.img_transform = img_transform if img_transform is not None else identical
        self.target_transform = target_transform if target_transform is not None else identical
        self.if_is_target = if_is_target

    def __call__(self, *imgs, random_seed=None) -> List[Union[Image.Image, torch.Tensor, np.ndarray]]:
        # assert cases
        imgs = imgs[0]  # imgs: Tuple[List]
        assert len(imgs) == len(self.if_is_target), f"len(imgs) should match len(if_is_target), " \
            f"given {len(imgs)} and {len(self.if_is_target)}."
        # assert cases ends
        random_seed: int = int(random.randint(0, 1e8)) if random_seed is None else int(random_seed)  # type ignore

        _imgs: List[Image.Image] = []
        for img, if_target in zip(imgs, self.if_is_target):
            with FixRandomSeed(random_seed):
                _img = self._transform(if_target)(img)
            _imgs.append(_img)
        return _imgs

    def _transform(self, is_target: bool) -> Callable:
        assert isinstance(is_target, bool)
        return self.img_transform if not is_target else self.target_transform
