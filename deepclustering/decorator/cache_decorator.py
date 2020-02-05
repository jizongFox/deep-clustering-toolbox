import inspect
from functools import update_wrapper
from multiprocessing import Manager

__all__ = ["SingleProcessCache", "MultiProcessCache"]


class SingleProcessCache:
    """
    >>> class A:
    >>>     @SingleProcessCache(key="index")
    >>>     def method(self,index):
    """

    def __init__(self, key=None) -> None:
        self._key = key
        self._is_class_method = False
        self._cache = self._initialize_cache()

    def _initialize_cache(self):
        return {}

    def _get_variable_from_keys(self, args, kwargs):
        assert self._key is not None
        if isinstance(self._key, (list, tuple)):
            result = []
            for k in self._key:
                r = self._get_variable_from_key(k, args, kwargs)
                result.append(r)
            return tuple(result)
        else:
            return self._get_variable_from_key(self._key, args, kwargs)

    def _get_variable_from_key(self, key, args, kwargs):
        # get the arguments and default values of the func
        assert (
            key in self.arg_list
        ), "key should be in the args list {}, given {}.".format(
            self.arg_list.args, key
        )
        # check if there is the key in the kwargs
        if key in kwargs:
            return kwargs[key]
        # check if there is the key in the args
        pos = self.arg_list.index(key)
        if pos < len(args):
            return args[pos]
        # the value is in the default setting.
        return self.default_dict[key]

    def get_key_value(self, args, kwargs):
        if self._key is None:
            if not self._is_class_method:
                _args = args + tuple(kwargs.items())
            else:
                _args = tuple(list(args)[1:]) + tuple(kwargs.items())
        else:
            _args = self._get_variable_from_keys(args, kwargs)
        return _args

    def __call__(self, func):
        func_args = inspect.getfullargspec(func)
        self.func = update_wrapper(self, func)
        self.func = func
        self.default_dict = {}
        if func_args.defaults:
            self.default_dict = dict(
                zip(func_args.args[::-1], func_args.defaults[::-1])
            )
        self.arg_list = func_args.args
        if "self" in self.arg_list:
            self._is_class_method = True
        if self._key is not None:
            if isinstance(self._key, (list, tuple)):
                for k in self._key:
                    assert k in self.arg_list
            else:
                assert self._key in self.arg_list

        def wrapper(*args, **kwargs):
            _args = self.get_key_value(args, kwargs)
            if _args in self._cache:
                return self._cache[_args]
            else:
                val = self.func(*args, **kwargs)
                self._cache[_args] = val
                return val

        return wrapper


class MultiProcessCache(SingleProcessCache):
    """
    >>> class A:
    >>>     @MultiProcessCache(key="index")
    >>>     def method(self,index):
    """

    def _initialize_cache(self):
        return Manager().dict()
