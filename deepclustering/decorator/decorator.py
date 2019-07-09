import _thread
import contextlib
import sys
import threading
import time
from functools import wraps
from threading import Thread

from torch.multiprocessing import Process


# in order to export functions
def export(fn):
    mod = sys.modules[fn.__module__]
    if hasattr(mod, "__all__"):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn


# in order to deal with BN tracking problem.
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


def threaded(_func=None, *, name="meter", daemon=True):
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


# in order to call a new process to play.
def processed(f):
    """Decorator to run the process in an extra process."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        func = Process(target=f, args=args, kwargs=kwargs)
        func.daemon = True
        func.start()
        return func

    return wrapper
