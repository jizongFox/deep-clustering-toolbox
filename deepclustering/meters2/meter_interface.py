from collections import OrderedDict
from typing import Dict, List, Optional

from .individual_meters._metric import _Metric

_Record_Type = Dict[str, float]


class MeterInteractMixin:
    individual_meters: Dict[str, _Metric]
    _ind_meter_dicts: Dict[str, _Metric]
    _group_dicts: Dict[str, List[str]]
    group: List[str]
    meter_names: List[str]

    def tracking_status(
        self, group_name=None, detailed_summary=False
    ) -> Dict[str, _Record_Type]:
        """
        return current training status from "ind_meters"
        :param group_name:
        :return:
        """
        if group_name:
            assert group_name in self.group
            return {
                k: v.detailed_summary() if detailed_summary else v.summary()
                for k, v in self.individual_meters.items()
                if k in self._group_dicts[group_name]
            }
        return {
            k: v.detailed_summary() if detailed_summary else v.summary()
            for k, v in self.individual_meters.items()
        }

    def add(self, meter_name, *args, **kwargs):
        assert meter_name in self.meter_names
        self._ind_meter_dicts[meter_name].add(*args, **kwargs)

    def reset(self) -> None:
        """
        reset individual meters
        :return: None
        """
        for v in self._ind_meter_dicts.values():
            v.reset()


class MeterInterface(MeterInteractMixin):
    """
    meter interface only concerns about the situation in one epoch,
    without considering historical record and save/load state_dict function.
    """

    def __init__(self) -> None:
        """
        :param meter_config: a dict of individual meter configurations
        """
        self._ind_meter_dicts: Dict[str, _Metric] = OrderedDict()
        self._group_dicts: Dict[str, List[str]] = OrderedDict()

    def __getitem__(self, meter_name: str) -> _Metric:
        try:
            return self._ind_meter_dicts[meter_name]
        except KeyError as e:
            print(f"meter_interface.meter_names:{self.meter_names}")
            raise e

    def register_meter(self, name: str, meter: _Metric, group_name=None) -> None:
        assert isinstance(name, str), name
        assert isinstance(
            meter, _Metric
        ), f"{meter.__class__.__name__} should be a subclass of {_Metric.__class__.__name__}, given {meter}."
        # add meters
        self._ind_meter_dicts[name] = meter
        if group_name is not None:
            if group_name not in self._group_dicts:
                self._group_dicts[group_name] = []
            self._group_dicts[group_name].append(name)

    def delete_meter(self, name: str) -> None:
        assert (
            name in self.meter_names
        ), f"{name} should be in `meter_names`: {self.meter_names}, given {name}."
        del self._ind_meter_dicts[name]
        for group, meter_namelist in self._group_dicts.items():
            if name in meter_namelist:
                meter_namelist.remove(name)

    def delete_meters(self, name_list: List[str]):
        assert isinstance(
            name_list, list
        ), f" name_list must be a list of str, given {name_list}."
        for name in name_list:
            self.delete_meter(name)

    @property
    def meter_names(self) -> List[str]:
        if hasattr(self, "_ind_meter_dicts"):
            return list(self._ind_meter_dicts.keys())

    @property
    def meters(self) -> Optional[Dict[str, _Metric]]:
        if hasattr(self, "_ind_meter_dicts"):
            return self._ind_meter_dicts

    @property
    def group(self) -> List[str]:
        return sorted(self._group_dicts.keys())

    @property
    def individual_meters(self):
        return self._ind_meter_dicts
