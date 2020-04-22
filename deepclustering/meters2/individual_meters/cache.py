import torch
from numbers import Number
from ._metric import _Metric
import numpy as np


class Cache(_Metric):
    """
    Cache is a meter to just store the elements in self.log. For statistic propose of use.
    """

    def __init__(self) -> None:
        super().__init__()
        self.log = []

    def reset(self):
        self.log = []

    def add(self, input):
        self.log.append(input)

    def value(self, **kwargs):
        return len(self.log)

    def summary(self) -> dict:
        return {"total elements": self.log.__len__()}

    def detailed_summary(self) -> dict:
        return self.summary()


class AveragewithStd(Cache):
    """
    this Meter is going to return the mean and std_lower, std_high for a list of scalar values
    """

    def add(self, input):
        assert (
            isinstance(input, Number)
            or (isinstance(input, torch.Tensor) and input.shape.__len__() <= 1)
            or (isinstance(input, np.ndarray) and input.shape.__len__() <= 1)
        )
        if torch.is_tensor(input):
            input = input.cpu().item()

        super().add(input)

    def value(self, **kwargs):
        return torch.Tensor(self.log).mean().item()

    def summary(self) -> dict:
        torch_log = torch.Tensor(self.log)
        mean = torch_log.mean()
        std = torch_log.std()
        return {
            "mean": mean.item(),
            "lstd": mean.item() - std.item(),
            "hstd": mean.item() + std.item(),
        }
