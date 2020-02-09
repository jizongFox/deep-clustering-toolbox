from abc import abstractmethod
from collections import OrderedDict
from typing import List, Dict, Any

import pandas as pd

REC_TYPE = Dict[int, Dict[str, float]]


class _Metric:
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

    @abstractmethod
    def value(self, **kwargs):
        pass

    @abstractmethod
    def summary(self) -> dict:
        pass

    @abstractmethod
    def detailed_summary(self) -> dict:
        pass

    @abstractmethod
    def get_plot_names(self) -> List[str]:
        pass

    def get_axe_numbers(self) -> int:
        return len(self.get_plot_names())


class _AggregatedMeter:
    """
    Aggregate historical information in a ordered dict.
    """

    def __init__(self) -> None:
        super().__init__()
        self._record_dict: REC_TYPE = OrderedDict()
        self._current_epoch: int = 0

    @property
    def record_dict(self) -> REC_TYPE:
        return self._record_dict

    @property
    def current_epoch(self) -> int:
        """ return current epoch
        """
        return self._current_epoch

    def summary(self) -> pd.DataFrame:
        # public interface of _AggregateMeter
        # todo: deal with the case where you have absent epoch
        validated_table = pd.DataFrame(self.record_dict).T
        if len(self.record_dict) < self.current_epoch:
            # you have the abscent value for the missing epoch
            missing_table = pd.DataFrame(
                index=set(range(self.current_epoch)) - set(self.record_dict.keys())
            )
            validated_table = validated_table.append(missing_table, sort=True)
        return validated_table

    def add(self, input_dict: Dict[str, float]) -> None:
        self._record_dict[self._current_epoch] = input_dict
        self._current_epoch += 1

    def reset(self) -> None:
        self._record_dict: REC_TYPE = OrderedDict()
        self._current_epoch = 0

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the class.
        """
        return self.__dict__

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): weight_scheduler state. Should be an object returned
                from a call to :math:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def __repr__(self):
        return str(pd.DataFrame(self.record_dict).T)
