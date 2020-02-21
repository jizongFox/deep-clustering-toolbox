import functools
import inspect
import os

from termcolor import colored

__all__ = ["lazy_load_checkpoint"]


def _extract_variable_from_args(func, args, name):
    variable_pos = inspect.getfullargspec(func).args.index(name)
    variable_value = None
    if len(args) >= variable_pos - 1:
        variable_value = args[variable_pos - 1]
        args = list(args)
        args[variable_pos - 1] = None
    return variable_value, tuple(args)


def _extract_variable_from_kwargs(kwargs, name):
    variable_value = kwargs.get(name, None)
    if variable_value:
        kwargs[name] = None
    return variable_value, kwargs


def lazy_load_checkpoint(func):
    """
    This function is to decorate the __init__ to get the checkpoint of the neural network.
    :param func:
    :return:
    """

    @functools.wraps(func)
    def wrapped_init_(self, *args, **kwargs):
        # check if args has "checkpoint_path" input:
        # if true, save checkpoint here and set it as None for the children method
        _checkpoint_path_from_args, args = _extract_variable_from_args(
            func, args, name="checkpoint_path"
        )
        # check if kwargs has "checkpoint_path" input:
        # if true, save checkpoint here and set it as None for the children method
        _checkpoint_path_from_kwarg, kwargs = _extract_variable_from_kwargs(
            kwargs, name="checkpoint_path"
        )
        _checkpoint_path = _checkpoint_path_from_args or _checkpoint_path_from_kwarg

        # perform normal __init__
        func(self, *args, **kwargs)
        # reset the checkpoint_path to the saved one.
        self._checkpoint = _checkpoint_path

        if self._checkpoint:
            try:
                self.load_checkpoint_from_path(self._checkpoint)
            except Exception as e:
                if os.environ.get("FORCE_LOAD_CHECKPOINT") == "1":
                    print(
                        colored(
                            f"!!!Loading checkpoint {self._checkpoint} failed with \n{e}."
                            f"\nDue to global environment variable `FORCE_LOAD_CHECKPOINT`=`1`, "
                            f"continue to train from scratch!",
                            "red",
                        )
                    )
                else:
                    raise e

        self.to(self._device)

    return wrapped_init_
