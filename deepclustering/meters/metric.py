import functools
from abc import abstractmethod
from typing import *

import pandas as pd
from easydict import EasyDict as edict

from deepclustering.decorator.decorator import export


def change_dataframe_name(dataframe: pd.DataFrame, name: str):
    dataframe.columns = list(map(lambda x: name + "_" + x, dataframe.columns))
    return dataframe


@export
class Metric(object):
    """Base class for all metrics.

    From: https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py
    """

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def add(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def value(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def summary(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def detailed_summary(self) -> dict:
        raise NotImplementedError


@export
class AggragatedMeter(object):
    """
    Aggregate historical information in a List.
    """

    def __init__(self) -> None:
        super().__init__()
        self.record: List[dict] = []

    # public interface of dict
    def summary(self, if_dict=False) -> Union[pd.DataFrame, List[dict]]:
        if if_dict:
            return self.record
        return pd.DataFrame(self.record)

    def add(self, input_dict) -> None:
        self.record.append(input_dict)

    def reset(self):
        self.record = []

    def state_dict(self) -> dict:
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != "meter"}

    def load_state_dict(self, state_dict) -> None:
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :math:`state_dict`.
        """
        self.__dict__.update(state_dict)


@export
class MeterInterface(object):
    """
    A listed of Aggregated Meters with names, that severs to be a interface for project.
    """

    def __init__(self, meter_config: Dict[str, Metric]) -> None:
        """
        :param meter_config: a dict of individual meter configurations
        """
        super().__init__()
        # check:
        for k, v in meter_config.items():
            assert isinstance(k, str), k
            assert isinstance(v, Metric), v  # can also check the subclasses.
        self.ind_meter_dict = (
            edict(meter_config) if not isinstance(meter_config, edict) else meter_config
        )
        for _, v in self.ind_meter_dict.items():
            v.reset()
        for k, v in self.ind_meter_dict.items():
            setattr(self, k, v)

        self.aggregated_meter_dict: Dict[str, AggragatedMeter] = edict(
            {k: AggragatedMeter() for k in self.ind_meter_dict.keys()}
        )

    def __getitem__(self, meter_name) -> Metric:
        return self.ind_meter_dict[meter_name]

    def register_new_meter(self, name: str, meter: Metric):
        assert isinstance(name, str), name
        assert isinstance(meter, Metric), meter
        self.ind_meter_dict[name] = meter
        self.aggregated_meter_dict[name] = AggragatedMeter()

    def summary(self) -> pd.DataFrame:
        """
        summary on the list of sub summarys, merging them together.
        :return:
        """
        list_of_summary = [
            change_dataframe_name(v.summary(), k)
            for k, v in self.aggregated_meter_dict.items()
        ]
        # merge the list
        summary = functools.reduce(
            lambda x, y: pd.merge(x, y, left_index=True, right_index=True),
            list_of_summary,
        )
        return pd.DataFrame(summary)

    def step(self) -> None:
        """
        This is to put individual Meter summary to Aggregated Meter dict
        And reset the individual Meters
        :return: None
        """
        for k in self.ind_meter_dict.keys():
            self.aggregated_meter_dict[k].add(self.ind_meter_dict[k].summary())
            self.ind_meter_dict[k].reset()

    @property
    def state_dict(self) -> dict:
        """
        to export dict
        :return: state dict
        """
        return {k: v.record for k, v in self.aggregated_meter_dict.items()}

    def load_state_dict(self, checkpoint):
        """
        to load dict
        :param checkpoint: dict
        :return:None
        """
        assert isinstance(checkpoint, dict)
        for k, v in self.aggregated_meter_dict.items():
            v.record = checkpoint[k]
        print(self.summary().tail())

    @classmethod
    def initialize_from_state_dict(cls, checkpoint: Dict[str, dict]):
        Meters = edict()
        submeter_names = list(checkpoint.keys())
        for k in submeter_names:
            Meters[k] = AggragatedMeter()
        wholeMeter = cls(
            names=submeter_names, listAggregatedMeter=list(Meters.values())
        )
        wholeMeter.load_state_dict(checkpoint)
        return wholeMeter, Meters
