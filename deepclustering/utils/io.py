from pathlib import Path, PosixPath
from pprint import pprint
from typing import Union, Dict, Any

import yaml


def path2Path(path):
    assert isinstance(path, (Path, str)), type(path)
    return Path(path) if isinstance(path, str) else path


def path2str(path):
    assert isinstance(path, (Path, str)), type(path)
    return str(path)


def yaml_load(yaml_path: Union[Path, str], verbose=False) -> Dict[str, Any]:
    """
    load yaml file given a file string-like file path. return must be a dictionary.
    :param yaml_path:
    :param verbose:
    :return:
    """
    assert isinstance(yaml_path, (Path, str, PosixPath)), type(yaml_path)
    with open(path2str(yaml_path), "r") as stream:
        data_loaded: dict = yaml.safe_load(stream)
    if verbose:
        print(f"Loaded yaml path:{str(yaml_path)}")
        pprint(data_loaded)
    return data_loaded


def write_yaml(
    dictionary: Dict, save_dir: Union[Path, str], save_name: str, force_overwrite=True
) -> None:
    save_path = path2Path(save_dir) / save_name
    if save_path.exists():
        if force_overwrite is False:
            save_name = (
                save_name.split(".")[0] + "_copy" + "." + save_name.split(".")[1]
            )
    with open(str(save_dir / save_name), "w") as outfile:  # type: ignore
        yaml.dump(dictionary, outfile, default_flow_style=False)


yaml_write = write_yaml
load_yaml = yaml_load
