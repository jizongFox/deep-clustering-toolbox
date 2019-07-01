from .metric import Metric


class Cache(Metric):
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
