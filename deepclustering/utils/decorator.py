import _thread
import contextlib
import sys
import time
from functools import wraps

from torch.multiprocessing import Process
from typing_inspect import is_union_type


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    # let the track_running_stats to be inverse
    model.apply(switch_attr)
    # return the model
    yield
    # let the track_running_stats to be inverse
    model.apply(switch_attr)


class Timer(object):
    """
    with Timer() as timer:
        ...
        ...
    print(timer.cost)
    ...
    """

    def __init__(self, start=None):
        self.start = start if start is not None else time.time()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop = time.time()
        self.cost = self.stop - self.start
        return exc_type is None


def export(fn):
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn


@export
def accepts(func):
    types = func.__annotations__
    for k, v in types.items():
        if is_union_type(v):
            types[k] = tuple(v.__args__)

    def check_accepts(*args, **kwargs):
        for (a, t) in zip(args, list(types.values())):
            assert isinstance(a, t), \
                "arg %r does not match %s" % (a, t)

        for k, v in kwargs.items():
            assert isinstance(v, types[k]), \
                f'kwargs {k}:{v} does not match {types[k]}'

        return func(*args, **kwargs)

    return check_accepts


def convert_params(f):
    """Decorator to call the process_params method of the class."""

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        return self.process_params(f, *args, **kwargs)

    return wrapper


def threaded(f):
    """Decorator to run the process in an extra thread."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        return _thread.start_new(f, args, kwargs)

    return wrapper


def processed(f):
    """Decorator to run the process in an extra process."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        func = Process(target=f, args=args, kwargs=kwargs)
        func.start()
        return func

    return wrapper
