import random
from typing import Callable, List

import numpy as np

from ..utils import identical


class FixRandomSeed(object):

    def __init__(self, random_seed: int = 0):
        self.random_seed = random_seed
        # backup
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
    """

    def __init__(self, img_transform: Callable = None, target_transform: Callable = None) -> None:
        super().__init__()
        self.img_transform = img_transform if img_transform is not None else identical
        self.target_transform = target_transform if target_transform is not None else identical

    def __call__(self, *imgs, if_is_target: List[bool] = [False, True, True]):
        def _transform(is_target: bool) -> Callable:
            assert isinstance(is_target, bool)
            return self.img_transform if not is_target else self.target_transform

        # assert cases
        imgs = imgs[0]  # imgs: Tuple[List]
        assert isinstance(imgs, list), \
            f"imgs provided must be a list or tuple, given {imgs}"
        # assert isinstance(imgs[0], Tensor)
        assert isinstance(if_is_target, (list, tuple)), \
            f"if_is_target must be a list or tuple, given {if_is_target}"
        assert isinstance(if_is_target[0], bool)
        assert len(imgs) == len(if_is_target), f"len(imgs) should match len(if_is_target), given {len(imgs)} " \
            f"and {len(if_is_target)}."
        # assert cases

        random_seed: int = int(random.randint(0, 1e8))

        _imgs = []
        for img, if_target in zip(imgs, if_is_target):
            with FixRandomSeed(random_seed):
                _img = _transform(if_target)(img)
            _imgs.append(_img)
        return _imgs
