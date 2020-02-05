# in this file, no dependency on the other module.
import collections
import os
import random
from copy import deepcopy as dcopy
from functools import partial
from functools import reduce
from math import isnan
from multiprocessing import Pool
from operator import and_
from typing import Iterable, Set, Tuple, TypeVar, Callable, List, Dict, Any, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from tqdm import tqdm

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", Tensor, np.ndarray)


class Identical(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, m):
        return m


def identical(x: Any) -> Any:
    """
    identical function
    :param x: function x
    :return: function x
    """
    return x


def set_nicer(nice) -> None:
    """
    set program priority
    :param nice: number to be set.
    :return: None
    """
    if nice:
        os.nice(nice)
        print(f"Process priority has been changed to {nice}.")


def set_environment(environment_dict: Dict[str, str] = None) -> None:
    if environment_dict:
        import os

        for k, v in environment_dict.items():
            os.environ[k] = str(v)
            print(f"setting environment {k}:{v}")


# reproducibility
def fix_all_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_benchmark(seed):
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# tqdm
class tqdm_(tqdm):
    def __init__(
        self,
        iterable=None,
        desc=None,
        total=None,
        leave=False,
        file=None,
        ncols=15,
        mininterval=0.1,
        maxinterval=10.0,
        miniters=None,
        ascii=None,
        disable=False,
        unit="it",
        unit_scale=False,
        dynamic_ncols=False,
        smoothing=0.3,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [" "{rate_fmt}{postfix}]",
        initial=0,
        position=None,
        postfix=None,
        unit_divisor=1000,
        write_bytes=None,
        gui=False,
        **kwargs,
    ):
        super().__init__(
            iterable,
            desc,
            total,
            leave,
            file,
            ncols,
            mininterval,
            maxinterval,
            miniters,
            ascii,
            disable,
            unit,
            unit_scale,
            dynamic_ncols,
            smoothing,
            bar_format,
            initial,
            position,
            postfix,
            unit_divisor,
            write_bytes,
            gui,
            **kwargs,
        )


# slack name for tqdm
class _tqdm(tqdm_):
    pass


# Assert utils
def uniq(a: Tensor) -> Set:
    """
    return unique element of Tensor
    Use python Optimized mode to skip assert statement.
    :rtype set
    :param a: input tensor
    :return: Set(a_npized)
    """
    return set([x.item() for x in a.unique()])


def sset(a: Tensor, sub: Iterable) -> bool:
    """
    if a tensor is the subset of the other
    :param a:
    :param sub:
    :return:
    """
    return uniq(a).issubset(sub)


def eq(a: Tensor, b: Tensor) -> bool:
    """
    if a and b are equal for torch.Tensor
    :param a:
    :param b:
    :return:
    """
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1) -> bool:
    """
    check if the matrix is the probability distribution
    :param t:
    :param axis:
    :return:
    """
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones, rtol=1e-4, atol=1e-4)


def one_hot(t: Tensor, axis=1) -> bool:
    """
    check if the Tensor is one hot.
    The tensor shape can be float or int or others.
    :param t:
    :param axis: default = 1
    :return: bool
    """
    return simplex(t, axis) and sset(t, [0, 1])


def intersection(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert a.dtype == torch.int, a.dtype
    assert b.dtype == torch.int, b.dtype
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])
    return a & b


def union(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])
    return a | b


def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
    assert simplex(probs, 1)
    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)

    return res


def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))

    b, *wh = seg.shape  # type:  Tuple[int, int, int]

    res: Tensor = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, *wh)
    assert one_hot(res)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, *_ = probs.shape
    assert simplex(probs)
    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)
    return res


def logit2one_hot(logit: Tensor) -> Tensor:
    probs = F.softmax(logit, 1)
    return probs2one_hot(probs)


# functions
def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))


def mmap_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    with Pool() as pool:
        return list(pool.map(fn, iter))


def uc_(fn: Callable) -> Callable:
    return partial(uncurry, fn)


def uncurry(fn: Callable, args: List[Any]) -> Any:
    return fn(*args)


def id_(x):
    return x


def assert_list(func: Callable[[A], bool], Iters: Iterable) -> bool:
    """
    List comprehensive assert for a function and a list of iterables.
    >>> assert assert_list(simplex, [torch.randn(2,10)]*10)
    :param func: assert function
    :param Iters:
    :return:
    """
    return reduce(and_, [func(x) for x in Iters])


def iter_average(input_iter: Iterable):
    return sum(input_iter) / len(input_iter)


# dictionary helper functions
def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# merge hierarchically two dictionaries
# todo: improve this function
def dict_merge(dct: Dict[str, Any], merge_dct: Dict[str, Any], re=True):
    """
    Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    # dct = dcopy(dct)
    if merge_dct is None:
        if re:
            return dct
        else:
            return
    for k, v in merge_dct.items():
        if (
            k in dct
            and isinstance(dct[k], dict)
            and isinstance(merge_dct[k], collections.Mapping)
        ):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]
    if re:
        return dcopy(dct)


# filter a flat dictionary with a lambda function
def dict_filter(
    dictionary: Dict[str, np.ndarray],
    filter_func: Callable = lambda k, v: (v != 0 and not isnan(v)),
):
    return {k: v for k, v in dictionary.items() if filter_func(k, v)}


# make a flatten dictionary to be printablely nice.
def nice_dict(input_dict: Dict[str, Union[int, float]]) -> str:
    """
    this function is to return a nice string to dictionary displace propose.
    :param input_dict: dictionary
    :return: string
    """
    assert isinstance(
        input_dict, dict
    ), f"{input_dict} should be a dict, given {type(input_dict)}."
    is_flat_dict = True
    for k, v in input_dict.items():
        if isinstance(v, dict):
            is_flat_dict = False
            break
    flat_dict = input_dict if is_flat_dict else flatten_dict(input_dict, sep="")
    string_list = [f"{k}:{v:.3f}" for k, v in flat_dict.items()]
    return ", ".join(string_list)


dict_flatten = flatten_dict
merge_dict = dict_merge
filter_dict = dict_filter


class Vectorize:
    r"""
    this class calls the np.vectorize with a mapping dict, in order to solve local memory share issue.
    """

    def __init__(self, mapping_dict: Dict[int, int]) -> None:
        super().__init__()
        self._mapping_dict = mapping_dict
        self._mapping_module = np.vectorize(lambda x: self._mapping_dict.get(x, 0))

    def __call__(self, np_tensor: np.ndarray):
        return self._mapping_module(np_tensor)

    def __repr__(self):
        return f"mapping_dict = {self._mapping_dict}"


def extract_from_big_dict(big_dict, keys) -> dict:
    """ Get a small dictionary with key in `keys` and value
        in big dict. If the key doesn't exist, give None.
        :param big_dict: A dict
        :param keys: A list of keys
    """
    #   TODO a bug has been found
    return {
        key: big_dict.get(key)
        for key in keys
        if big_dict.get(key, "not_found") != "not_found"
    }


# meta function for interface
def _register(
    name: str, callable: Callable, alias=None, CALLABLE_DICT: dict = {}
) -> None:
    """ Private method to register the architecture to the ARCH_CALLABLES
        :param name: A str
        :param callable: The callable that return the nn.Module
        :param alias: None, or a list of string, or str
    """
    if name in CALLABLE_DICT:
        raise ValueError("{} already exists!".format(name.lower()))
    CALLABLE_DICT[name.lower()] = callable
    if alias:
        if isinstance(alias, str):
            alias = [alias]
        for other_arch in alias:
            if other_arch.lower() in CALLABLE_DICT:
                raise ValueError(
                    "alias {} for {} already exists!".format(
                        other_arch.lower(), name.lower()
                    )
                )
            CALLABLE_DICT[other_arch.lower()] = callable
