import argparse
from copy import deepcopy as dcopy
from functools import reduce
from pprint import pprint
from typing import Union, List, Dict, Any

import yaml
from pathlib2 import Path, PosixPath

from .general import map_, dict_merge

__all__ = ["yaml_parser", "yaml_load"]

D = Dict[str, Any]


# todo, add type check
# argparser
def yaml_load(yaml_path: Union[Path, str], verbose=False) -> dict:
    assert isinstance(yaml_path, (Path, str, PosixPath)), type(yaml_path)
    with open(str(yaml_path), 'r') as stream:
        data_loaded: dict = yaml.safe_load(stream)
    if verbose:
        print(f'Loaded yaml path:{str(yaml_path)}')
        pprint(data_loaded)
    return data_loaded


def yaml_parser(verbose=False) -> Dict[str, D]:
    parser = argparse.ArgumentParser('Augment parser for yaml config')
    parser.add_argument('strings', nargs='*', type=str, default=[''])

    args: argparse.Namespace = parser.parse_args()
    args: dict = _parser(args.strings)  # type: ignore
    if not args:
        args = {}
    if verbose:
        print('Argparsed args:')
        pprint(args)
    return args


def _parser(strings: List[str]) -> List[dict]:
    assert isinstance(strings, list)
    ## no doubled augments
    assert set(map_(lambda x: x.split('=')[0], strings)).__len__() == strings.__len__(), 'Augment doubly input.'
    args: List[dict] = [_parser_(s) for s in strings]
    args = reduce(lambda x, y: dict_merge(x, y, True), args)  # type: ignore
    return args


def _parser_(input_string: str) -> Union[dict, None]:
    if input_string.__len__() == 0:
        return None
    assert input_string.find('=') > 0, f"Input args should include '=' to include the value"
    keys, value = input_string.split('=')[:-1][0].replace(' ', ''), input_string.split('=')[1].replace(' ', '')
    keys = keys.split('.')
    keys.reverse()
    for k in keys:
        d = {}
        d[k] = value
        value = dcopy(d)
    return dict(value)
