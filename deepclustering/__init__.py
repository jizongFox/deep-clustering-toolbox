import subprocess
from enum import Enum
from pathlib import Path

DATA_PATH = str(Path(__file__).parents[1] / ".data")
PROJECT_PATH = str(Path(__file__).parents[1])
Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
CC_wrapper_path = str(Path(PROJECT_PATH) / "deepclustering/utils/CC_wrapper.sh")
LC_wrapper_path = str(Path(PROJECT_PATH) / "deepclustering/utils/LOCAL_wrapper.sh")
JA_wrapper_path = str(Path(PROJECT_PATH) / "deepclustering/utils/JOBARRAY_wrapper.sh")
try:
    __git_hash__ = (
        subprocess.check_output([f"cd {PROJECT_PATH}; git rev-parse HEAD"], shell=True)
        .strip()
        .decode()
    )
except:
    __git_hash__ = None


class ModelMode(Enum):
    """ Different mode of model """

    TRAIN = "TRAIN"  # during training
    EVAL = "EVAL"  # eval mode. On validation data
    PRED = "PRED"

    @staticmethod
    def from_str(mode_str):
        """ Init from string
            :param mode_str: ['train', 'eval', 'predict']
        """
        if mode_str == "train":
            return ModelMode.TRAIN
        elif mode_str == "eval":
            return ModelMode.EVAL
        elif mode_str == "predict":
            return ModelMode.PRED
        else:
            raise ValueError("Invalid argument mode_str {}".format(mode_str))
