from ._metric import _Metric


# this meter is to show the instance value, instead of print.


class InstanceValue(_Metric):
    def __init__(self) -> None:
        super().__init__()
        self.instance_value = None

    def reset(self):
        self.instance_value = None

    def add(self, value):
        self.instance_value = value

    def value(self, **kwargs):
        return self.instance_value

    def summary(self) -> dict:
        return {"value": self.instance_value}

    def detailed_summary(self) -> dict:
        return {"value": self.instance_value}
