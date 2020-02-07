from pathlib import Path, PosixPath
from pprint import pprint
from typing import Union, Dict, Any

import yaml


def yaml_load(yaml_path: Union[Path, str], verbose=False) -> Dict[str, Any]:
    """
    load yaml file given a file string-like file path. return must be a dictionary.
    :param yaml_path:
    :param verbose:
    :return:
    """
    assert isinstance(yaml_path, (Path, str, PosixPath)), type(yaml_path)
    with open(str(yaml_path), "r") as stream:
        data_loaded: dict = yaml.safe_load(stream)
    if verbose:
        print(f"Loaded yaml path:{str(yaml_path)}")
        pprint(data_loaded)
    return data_loaded


load_yaml = yaml_load


def write_yaml(dictionary: Dict, save_dir: Union[Path, str], save_name: str) -> None:
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    with open(str(save_dir / save_name), "w") as outfile:  # type: ignore
        yaml.dump(dictionary, outfile, default_flow_style=False)


yaml_write = write_yaml
