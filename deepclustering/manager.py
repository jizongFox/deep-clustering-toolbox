import warnings
from pathlib import Path
from pprint import pprint
from typing import Dict, Any

from deepclustering.utils import yaml_parser, yaml_load, dict_merge


class ConfigManger:
    DEFAULT_CONFIG = ""

    def __init__(self, DEFAULT_CONFIG_PATH: str = None, verbose=True, integrality_check=True) -> None:
        self.parsed_args: Dict[str, Any] = yaml_parser(verbose=verbose)
        if DEFAULT_CONFIG_PATH is None:
            warnings.warn('No default yaml is provided, only used for parser input arguments.', UserWarning)
            return
        self.SET_DEFAULT_CONFIG_PATH(DEFAULT_CONFIG_PATH)
        self.default_config: Dict[str, Any] = yaml_load(self.parsed_args.get('Config', self.DEFAULT_CONFIG),
                                                        verbose=verbose)
        self.merged_config: Dict[str, Any] = dict_merge(self.default_config, self.parsed_args, re=True)
        if integrality_check:
            self._check_integrality(self.merged_config)
        if verbose:
            print("Merged args:")
            pprint(self.merged_config)

    @classmethod
    def SET_DEFAULT_CONFIG_PATH(cls, default_config_path: str):
        path: Path = Path(default_config_path)
        assert path.exists(), path
        assert path.is_file(), path
        assert path.with_suffix('.yaml') or path.with_suffix('.yml')
        cls.DEFAULT_CONFIG = str(default_config_path)

    @staticmethod
    def _check_integrality(merged_dict=Dict[str, Any]):
        assert merged_dict.get('Arch'), f"Merged dict integrity check failed,{merged_dict.keys()}"
        assert merged_dict.get('Optim'), f"Merged dict integrity check failed,{merged_dict.keys()}"
        assert merged_dict.get('Scheduler'), f"Merged dict integrity check failed,{merged_dict.keys()}"
        assert merged_dict.get('Trainer'), f"Merged dict integrity check failed,{merged_dict.keys()}"

    @property
    def config(self):
        return self.merged_config
