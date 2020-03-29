from typing import Union, List

from deepclustering.meters import MeterInterface
from deepclustering.model import Model
from deepclustering.schedulers.customized_scheduler import WeightScheduler
from deepclustering.schedulers.lr_scheduler import _LRScheduler
from ._hooks import HookBase


class SchedulerHook(HookBase):
    _schedulers: List

    def after_train_epoch(self, *args, **kwargs):
        for scheduler in self._schedulers:
            scheduler.step()

    def before_train(self, *args, **kwargs):
        for scheduler in self._schedulers:
            if hasattr(scheduler, "reset"):
                scheduler.reset()


class LrSchedulerHook(SchedulerHook):
    def __init__(self, schedulers: List[Union[_LRScheduler, Model]]) -> None:
        super().__init__()
        self._schedulers: List[_LRScheduler] = []
        for scheduler in schedulers:
            if isinstance(scheduler, _LRScheduler):
                self._schedulers.append(scheduler)
            elif isinstance(scheduler, Model):
                self._schedulers.append(scheduler._scheduler)
            else:
                raise TypeError(type(scheduler))


class WeightSchedulerHook(SchedulerHook):
    def __init__(self, schedulers: List[WeightScheduler]) -> None:
        super().__init__()
        self._schedulers: List[WeightScheduler] = []
        for scheduler in schedulers:
            if isinstance(scheduler, WeightScheduler):
                self._schedulers.append(scheduler)
            else:
                raise TypeError(type(scheduler))


class MeterSchedulerHook(SchedulerHook):
    def __init__(self, meters: List[MeterInterface]) -> None:
        super().__init__()
        self._schedulers: List[MeterInterface] = []
        for meter in meters:
            if isinstance(meter, MeterInterface):
                self._schedulers.append(meter)
            else:
                raise TypeError(type(meter))
