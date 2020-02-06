import functools
from abc import abstractmethod
from collections import OrderedDict
from typing import List, Dict, Any

import pandas as pd
from easydict import EasyDict as edict

from ._utils import rename_df_columns

__all__ = ["MeterInterface"]

# todo:
"""
we should incorporate the `Meters` and `Drawers` in the same interace, instead of setting them separately.
"""
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


class MeterInterface:
    """
    Interface for meters for each batch and epoch
    """

    def __init__(self, meter_config: Dict[str, _Metric]) -> None:
        """
        :param meter_config: a dict of individual meter configurations
        """
        # check input meter configurations:
        for k, v in meter_config.items():
            assert isinstance(k, str), k
            assert issubclass(
                type(v), _Metric
            ), f"{v.__class__.__name__} should be a subclass of {_Metric.__class__.__name__}, given {v}."  # can also check the subclasses.
        self._ind_meter_dicts: Dict[str, _Metric] = edict(
            meter_config
        ) if not isinstance(meter_config, edict) else meter_config

        for v in self._ind_meter_dicts.values():
            v.reset()
        for k, v in self._ind_meter_dicts.items():
            setattr(self, k, v)

        self._aggregated_meter_dicts: Dict[str, _AggregatedMeter] = edict(
            {k: _AggregatedMeter() for k in self._ind_meter_dicts.keys()}
        )

    def __getitem__(self, meter_name: str) -> _Metric:
        try:
            return self._ind_meter_dicts[meter_name]
        except AttributeError as e:
            raise e

    def register_new_meter(self, name: str, meter: _Metric) -> None:
        assert isinstance(name, str), name
        assert issubclass(
            type(meter), _Metric
        ), f"{meter.__class__.__name__} should be a subclass of {_Metric.__class__.__name__}, given {meter}."

        # add meters
        self._ind_meter_dicts[name] = meter
        setattr(self, name, meter)
        self._aggregated_meter_dicts[name] = _AggregatedMeter()
        self._aggregated_meter_dicts[name]._current_epoch = self.current_epoch

    def delete_meter(self, name: str) -> None:
        assert (
            name in self.meter_names
        ), f"{name} should be in `meter_names`: {self.meter_names}, given {name}."
        del self._ind_meter_dicts[name]
        del self._aggregated_meter_dicts[name]
        delattr(self, name)

    def delete_meters(self, name_list: List[str]):
        assert isinstance(
            name_list, list
        ), f" name_list must be a list of str, given {name_list}."
        for name in name_list:
            self.delete_meter(name)

    def step(self, detailed_summary=False) -> None:
        """
        This is to put individual Meter summary to Aggregated Meter dict and reset the individual Meters
        this is supposed to be called at the end of an epoch.
        :param detailed_summary: return `detailed_summary` instead of `summary`
        :return: None
        """
        for k in self.meter_names:
            self._aggregated_meter_dicts[k].add(
                self._ind_meter_dicts[k].summary()
                if not detailed_summary
                else self._ind_meter_dicts[k].detailed_summary()
            )
        self.reset_before_epoch()

    def reset_before_epoch(self) -> None:
        """
        reset individual meters
        :return: None
        """
        for v in self._ind_meter_dicts.values():
            v.reset()

    def clear_history(self) -> None:
        """This is to clear the aggregated meters for history"""
        for v in self._aggregated_meter_dicts.values():
            v.reset()

    def reset_all(self):
        """
        This is to call at the
        :return:
        """
        self.reset_before_epoch()
        self.clear_history()

    def reset(self):
        self.reset_all()

    def state_dict(self) -> dict:
        """
        to export dict
        :return: state dict
        """
        return {k: v.state_dict() for k, v in self._aggregated_meter_dicts.items()}

    def load_state_dict(self, checkpoint):
        """
        to load dict
        :param checkpoint: dict
        :return:None
        """
        assert isinstance(checkpoint, dict)
        _old_keys = checkpoint.keys()
        _new_keys = self.state_dict().keys()

        missed_keys = list(set(_new_keys) - set(_old_keys))
        redundant_keys = list(set(_old_keys) - set(_new_keys))
        if missed_keys.__len__() > 0:
            print(f"Found missed keys: {', '.join(missed_keys)}")
        if redundant_keys.__len__() > 0:
            print(f"Found redundant keys: {', '.join(redundant_keys)}")

        for k, v in self._aggregated_meter_dicts.items():
            try:
                v.load_state_dict(checkpoint[k])
            except KeyError:
                # you have a missed key,just leave it alone and set the current_epoch to align others.
                v._current_epoch = self.current_epoch
        current_epoch = self._aggregated_meter_dicts[self.meter_names[0]].current_epoch
        for k in self.meter_names:
            assert current_epoch == self._aggregated_meter_dicts[k].current_epoch
        print(self.summary())

    @property
    def meter_names(self) -> List[str]:
        return list(self._aggregated_meter_dicts.keys())

    @property
    def current_epoch(self):
        return self._aggregated_meter_dicts[self.meter_names[0]].current_epoch

    def summary(self) -> pd.DataFrame:
        """
        summary on the list of sub summarys, merging them together.
        :return:
        """
        list_of_summary = [
            rename_df_columns(v.summary(), k)
            for k, v in self._aggregated_meter_dicts.items()
        ]
        # merge the list
        summary = functools.reduce(
            lambda x, y: pd.merge(x, y, left_index=True, right_index=True),
            list_of_summary,
        )
        return pd.DataFrame(summary)
