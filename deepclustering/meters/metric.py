import functools
from typing import List, Iterable, Union, Optional

import pandas as pd

from ..utils import export


def change_dataframe_name(dataframe: pd.DataFrame, name: str):
    dataframe.columns = list(map(lambda x: name + '_' + x, dataframe.columns))
    return dataframe


@export
class Metric(object):
    """Base class for all metrics.

    From: https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py
    """

    def reset(self):
        pass

    def add(self, **kwargs):
        pass

    def value(self, **kwargs):
        pass

    def summary(self) -> dict:
        raise NotImplementedError

    def detailed_summary(self) -> dict:
        raise NotImplementedError


@export
class AggragatedMeter(object):
    '''
    Aggragate historical information in a List.
    '''

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
        return {key: value for key, value in self.__dict__.items() if key != 'meter'}

    def load_state_dict(self, state_dict) -> None:
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)


@export
class ListAggregatedMeter(object):

    def __init__(self,
                 listAggregatedMeter: List[AggragatedMeter],
                 names: Iterable[str] = None
                 ) -> None:
        super().__init__()
        self.ListAggragatedMeter: List[AggragatedMeter] = listAggregatedMeter
        self.names = names
        assert self.ListAggragatedMeter.__len__() == self.names.__len__()
        assert isinstance(self.ListAggragatedMeter, list), type(self.ListAggragatedMeter)

    def __getitem__(self, index: int):
        return self.ListAggragatedMeter[index]

    def summary(self) -> pd.DataFrame:
        '''
        summary on the list of subsummarys, merging them together.
        :return:
        '''

        list_of_summary = [change_dataframe_name(self.ListAggragatedMeter[i].summary(), n) \
                           for i, n in enumerate(self.names)]

        summary = functools.reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), list_of_summary)

        return pd.DataFrame(summary)

    @property
    def state_dict(self):
        return {n: l.record for n, l in zip(self.names, self.ListAggragatedMeter)}

    def load_state_dict(self, checkpoint):
        assert isinstance(checkpoint, dict)
        for n, l in zip(self.names, self.ListAggragatedMeter):
            l.record = checkpoint[n]
        print(self.summary())
