# todo: this file is a mixin class to help with some other classes
from abc import abstractmethod

from ..model import Model


# I think this should be something like mixin class for multiinherent class


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

    @abstractmethod
    def update(self, *args, **kwargs):
        """
        This is the main loop for updating the method algorithm within a minibatch, need to override.
        :param args:
        :param kwargs:
        :return:
        """

    @property
    def state_dict(self):
        # todo: it can be problematic when pass model to dict.
        return self.__dict__

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
