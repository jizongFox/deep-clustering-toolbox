import warnings
from ..utils import set_nicer
from copy import deepcopy as dcp


def _warnings(args: tuple, kwargs: dict):
    # change priority
    kwargs = dcp(kwargs)
    set_nicer(kwargs.get("nice"))
    kwargs.pop("nice", None)

    if len(args) > 0:
        warnings.warn(f"Received unassigned args with args: {args}.")
    if len(kwargs) > 0:
        kwarg_str = ", ".join([f"{k}:{v}" for k, v in kwargs.items()])
        warnings.warn(f"Received unassigned kwargs: \n{kwarg_str}")
