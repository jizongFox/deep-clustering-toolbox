__all__ = ["lazy_load_checkpoint"]
import functools
import inspect
import os

from termcolor import colored


def _extract_checkpoint_path_from_args(func, args):
    checkpoint_path_pos = inspect.getfullargspec(func).args.index("checkpoint_path")
    _checkpoint_path = None
    if len(args) >= checkpoint_path_pos - 1:
        _checkpoint_path = args[checkpoint_path_pos - 1]
        args = list(args)
        args[checkpoint_path_pos - 1] = None
    return _checkpoint_path, tuple(args)


def _extract_checkpoint_path_from_kwargs(kwargs):
    _checkpoint_path = kwargs.get("checkpoint_path")
    if _checkpoint_path:
        kwargs["checkpoint_path"] = None
    return _checkpoint_path, kwargs


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
        _checkpoint_path_from_args, args = _extract_checkpoint_path_from_args(
            func, args
        )
        # check if kwargs has "checkpoint_path" input:
        # if true, save checkpoint here and set it as None for the children method
        _checkpoint_path_from_kwarg, kwargs = _extract_checkpoint_path_from_kwargs(
            kwargs
        )
        _checkpoint_path = _checkpoint_path_from_args or _checkpoint_path_from_kwarg

        # perform normal __init__
        func(self, *args, **kwargs)
        # reset the checkpoint_path to the saved one.
        self.checkpoint = _checkpoint_path

        if self.checkpoint:
            try:
                self.load_checkpoint_from_path(self.checkpoint)
            except Exception as e:
                if os.environ.get("FORCE_LOAD_CHECKPOINT") == "1":
                    print(
                        colored(
                            f"!!!Loading checkpoint {self.checkpoint} failed with \n{e}."
                            f"\nDue to global environment variable `FORCE_LOAD_CHECKPOINT`=`1`, continue to train from scratch!",
                            "red",
                        )
                    )
                else:
                    raise e

        self.to(self.device)

    return wrapped_init_
