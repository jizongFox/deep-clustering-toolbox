import inspect
import sys

from typing_inspect import is_union_type
import contextlib
from .general import one_hot

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    # let the track_running_stats to be inverse
    model.apply(switch_attr)
    yield
    # let the track_running_stats to be inverse
    model.apply(switch_attr)



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


@export
def onehot(name):
    assert isinstance(name, (str, list))

    def check_onehot(f):
        f_sig = inspect.signature(f)
        if isinstance(name, str):
            assert name in f_sig.parameters.keys()
        else:
            assert set(f_sig.parameters.keys()).issuperset(
                set(name)), f'{name} should be included in {list(f_sig.parameters.keys())}'

        def new_f(*args, **kwds):
            for (a, t) in zip(args, f_sig.parameters.keys()):
                if t == name or t in name:
                    assert one_hot(a, 1), f'{t}={a} onehot check failed'

            for k, v in kwds.items():
                if k == name or k in name:
                    assert one_hot(v, 1), f'{k}={v} onehot check failed'
            return f(*args, **kwds)

        new_f.__name__ = f.__name__
        return new_f

    return check_onehot
