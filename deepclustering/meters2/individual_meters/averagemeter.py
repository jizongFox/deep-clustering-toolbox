from typing import List

import numpy as np

from ._metric import _Metric


class AverageValueMeter(_Metric):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = 0.0

    def summary(self) -> dict:
        # this function returns a dict and tends to aggregate the historical results.
        return {"mean": self.value()[0]}

    def detailed_summary(self) -> dict:
        # this function returns a dict and tends to aggregate the historical results.
        return {"mean": self.value()[0], "val": self.value()[1]}

    def __repr__(self):
        def _dict2str(value_dict: dict):
            return "\t".join([f"{k}:{v}" for k, v in value_dict.items()])

        return f"{self.__class__.__name__}: n={self.n} \n \t {_dict2str(self.detailed_summary())}"

    def get_plot_names(self) -> List[str]:
        return ["mean"]
