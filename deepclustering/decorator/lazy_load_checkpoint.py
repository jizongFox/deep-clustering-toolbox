__all__ = ["lazy_load_checkpoint"]
import functools
import os

from termcolor import colored


def lazy_load_checkpoint(func):
    """
    This function is to decorate the __init__ to get the checkpoint of the neural network.
    :param func:
    :return:
    """

    @functools.wraps(func)
    def wrapped_init_(self, *args, **kwargs):
        _checkpoint_path = kwargs.get("checkpoint_path")
        # setting all parent class with checkpoint=None
        if _checkpoint_path:
            kwargs["checkpoint_path"] = None
        # perform normal __init__
        func(self, *args, **kwargs)
        # reset the checkpoint_path
        self.checkpoint = _checkpoint_path

        if self.checkpoint:
            try:
                self.load_checkpoint_from_path(self.checkpoint)
            except Exception as e:
                if os.environ.get("FORCE_LOAD_CHECKPOINT") == "1":
                    print(colored(
                        f"!!!Loading checkpoint {self.checkpoint} failed with \n{e}."
                        f"\nDue to global environment variable `FORCE_LOAD_CHECKPOINT`=`1`, continue to train from scratch!",
                        "red"))
                else:
                    raise e

        self.to(self.device)

    return wrapped_init_
