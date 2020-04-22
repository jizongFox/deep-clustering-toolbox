from ._metric import _Metric

# individual package for meters based on
"""
>>>class _Metric(metaclass=ABCMeta):
>>>    @abstractmethod
>>>    def reset(self):
>>>        pass
>>>
>>>    @abstractmethod
>>>    def add(self, *args, **kwargs):
>>>        pass
>>>
>>>    @abstractmethod
>>>    def log(self):
>>>        pass
>>>
>>>    @abstractmethod
>>>    def summary(self) -> dict:
>>>        pass
>>>
>>>    @abstractmethod
>>>    def detailed_summary(self) -> dict:
>>>        pass
"""

from .averagemeter import AverageValueMeter
from .confusionmatrix import ConfusionMatrix
from .hausdorff import HaussdorffDistance
from .instance import InstanceValue
from .iou import IoU
