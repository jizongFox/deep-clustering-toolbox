from abc import ABCMeta
from collections import OrderedDict
from typing import Dict, OrderedDict as OrderedDict_Type, Any, Union

import pandas as pd

_Record_Type = Dict[str, float]
_Save_Type = OrderedDict_Type[int, _Record_Type]

__all__ = ["HistoricalContainer"]


class HistoricalContainer(metaclass=ABCMeta):
    """
    Aggregate historical information in a ordered dict.
    """

    def __init__(self) -> None:
        self._record_dict: _Save_Type = OrderedDict()
        self._current_epoch: int = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    def trainer(self):
        return self._trainer

    @property
    def record_dict(self) -> _Save_Type:
        return self._record_dict

    def get_record_dict(self, epoch=None) -> Union[_Record_Type, _Save_Type]:
        if epoch is None:
            return self.record_dict
        assert epoch in self._record_dict.keys(), "epoch {} not saved in {}".format(
            epoch, ", ".join(list(self._record_dict.keys()))
        )
        return self.record_dict[epoch]

    @property
    def current_epoch(self) -> int:
        """ return current epoch
        """
        return self._current_epoch

    def summary(self) -> pd.DataFrame:
        # todo: deal with the case where you have absent epoch
        validated_table = pd.DataFrame(self.record_dict).T
        # check if having missing values
        if len(self.record_dict) < self.current_epoch:
            missing_table = pd.DataFrame(
                index=set(range(self.current_epoch)) - set(self.record_dict.keys())
            )
            validated_table = validated_table.append(missing_table, sort=True)
        return validated_table

    def add(self, input_dict: _Record_Type, epoch=None) -> None:
        if epoch:
            self._current_epoch = epoch
        self._record_dict[self._current_epoch] = input_dict
        self._current_epoch += 1

    def reset(self) -> None:
        self._record_dict: _Save_Type = OrderedDict()
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
