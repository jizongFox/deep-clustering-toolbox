import _thread
import contextlib
import random
import sys
import threading
import time
from functools import wraps
from threading import Thread

import numpy as np
from torch import nn
from torch.multiprocessing import Process


# in order to export functions
def export(fn):
    mod = sys.modules[fn.__module__]
    if hasattr(mod, "__all__"):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn


def _extract_bn_modules(model):
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]


# in order to deal with BN tracking problem.
# this only works for PyTorch<=1.1.0
@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, "track_running_stats"):
            m.track_running_stats ^= True

    # let the track_running_stats to be inverse
    model.apply(switch_attr)
    # return the model
    yield
    # let the track_running_stats to be inverse
    model.apply(switch_attr)


class _disable_tracking_bn_stats_pytoch_el_1_1_0:
    """
    This is to deal with the bug linked with track_statistic_bn=False, see: https://github.com/perrying/realistic-ssl-evaluation-pytorch/issues/3
    """

    def __init__(self, model):
        self.bn_modules = _extract_bn_modules(model)
        self.moments = [m.momentum for m in self.bn_modules]

    def __enter__(self):
        for m in self.bn_modules:
            m.momentum = 0

    def __exit__(self, exc_type, exc_val, exc_tb):
        for module, momentum in zip(self.bn_modules, self.moments):
            module.momentum = momentum


# in order to count execution time
class TimeBlock:
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


def timethis(func):
    """
    Decorator that reports the execution time.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, end - start)
        return result

    return wrapper


# in order to convert parameter types
def convert_params(f):
    """Decorator to call the process_params method of the class."""

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        return self.process_params(f, *args, **kwargs)

    return wrapper


# in order to begin a new thread for IO-bounded job.
def threaded_(f):
    """Decorator to run the process in an extra thread."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        return _thread.start_new(f, args, kwargs)

    return wrapper


def threaded(_func=None, *, name="meter", daemon=False):
    """Decorator to run the process in an extra thread."""

    def decorator_thread(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            new_thread = Thread(target=f, args=args, kwargs=kwargs, name=name)
            new_thread.daemon = daemon
            new_thread.start()
            return new_thread

        return wrapper

    if _func is None:
        return decorator_thread
    else:
        return decorator_thread(_func)


class WaitThreadsEnd:
    def __init__(self, thread_name: str = "meter") -> None:
        super().__init__()
        self.thread_name = thread_name

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        for t in threading.enumerate():
            if t.name == self.thread_name:
                t.join()

    def __call__(self, func):
        @wraps(func)
        def decorate_func(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate_func


wait_thread_ends = WaitThreadsEnd


# in order to call a new process to play.
def processed(f):
    """Decorator to run the process in an extra process."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        func = Process(target=f, args=args, kwargs=kwargs)
        func.daemon = False
        func.start()
        return func

    return wrapper


class FixRandomSeed:
    """
    This class fixes the seeds for numpy and random pkgs.
    """

    def __init__(self, random_seed: int = 0):
        self.random_seed = random_seed
        self.randombackup = random.getstate()
        self.npbackup = np.random.get_state()

    def __enter__(self):
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def __exit__(self, *_):
        np.random.set_state(self.npbackup)
        random.setstate(self.randombackup)
