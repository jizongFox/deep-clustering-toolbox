from abc import abstractmethod, ABCMeta


class _Metric(metaclass=ABCMeta):
    """Base class for all metrics.
    record the values within a single epoch
    From: https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py
    """

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def add(self, *args, **kwargs):
        pass

    # @abstractmethod
    # def log(self):
    #     pass

    @abstractmethod
    def summary(self) -> dict:
        pass

    @abstractmethod
    def detailed_summary(self) -> dict:
        pass
