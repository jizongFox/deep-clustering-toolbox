import warnings
from abc import abstractmethod

from ..model import Model


class _Method(object):
    """
    This is the meta class used for Method, plugin class. It can be used as the meta class for ADMM,
    subspace clustering etc.
    """

    def __init__(self, model: Model, *args, **kwargs):
        """
        internal class method
        :param model: type of Model
        :param args: unassigned args
        :param kwargsa: unassigned kwargs
        """
        # warning control
        if len(args) > 0:
            warnings.warn(f'Received unassigned args with args: {args}.')
        if len(kwargs) > 0:
            kwarg_str = ", ".join([f"{k}:{v}" for k, v in kwargs.items()])
            warnings.warn(f'Received unassigned kwargs: \n{kwarg_str}')
        # warning control ends
        self.model = model

    @abstractmethod
    def set_input(self, *args, **kwargs):
        """
        set mini-batch
        :param minibatch_input:
        :param args:
        :param kwargs:
        :return:
        """
        # warning control
        if len(args) > 0:
            warnings.warn(f'Received unassigned args with args: {args}.')
        if len(kwargs) > 0:
            kwarg_str = ", ".join([f"{k}:{v}" for k, v in kwargs.items()])
            warnings.warn(f'Received unassigned kwargs: \n{kwarg_str}')
        # warning control ends

    @abstractmethod
    def update(self, *args, **kwargs):
        """
        This is the main loop for updating the method algorithm within a minibatch, need to override.
        :param args:
        :param kwargs:
        :return:
        """
        # warning control
        if len(args) > 0:
            warnings.warn(f'Received unassigned args with args: {args}.')
        if len(kwargs) > 0:
            kwarg_str = ", ".join([f"{k}:{v}" for k, v in kwargs.items()])
            warnings.warn(f'Received unassigned kwargs: \n{kwarg_str}')
        # warning control ends
