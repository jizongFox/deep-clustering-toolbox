from typing import List

import torch
from sklearn.metrics import cohen_kappa_score
from torch import Tensor

from ._metric import _Metric


class KappaMetrics(_Metric):
    """ SKLearnMetrics computes various classification metrics at the end of a batch.
     Unforunately, doesn't work when used with generators...."""

    def __init__(self) -> None:
        super().__init__()
        self.kappa = []

    def add(
        self, predicts: List[Tensor], target: Tensor, considered_classes: List[int]
    ):
        for predict in predicts:
            assert predict.shape == target.shape
        predicts = [predict.detach().data.cpu().numpy().ravel() for predict in predicts]
        target = target.detach().data.cpu().numpy().ravel()
        mask = [t in considered_classes for t in target]
        predicts = [predict[mask] for predict in predicts]
        target = target[mask]
        kappa_score = [cohen_kappa_score(predict, target) for predict in predicts]
        self.kappa.append(kappa_score)

    def reset(self):
        self.kappa = []

    def value(self):
        return torch.Tensor(self.kappa).mean(0)

    def summary(self):
        return {f"kappa{i}": self.value()[i].item() for i in range(len(self.value()))}

    def detailed_summary(self):
        return {f"kappa{i}": self.value()[i].item() for i in range(len(self.value()))}


class Kappa2Annotator(KappaMetrics):
    def __init__(self) -> None:
        super().__init__()

    def add(
        self,
        predict1: Tensor,
        predict2: Tensor,
        gt=Tensor,
        considered_classes=[1, 2, 3],
    ):
        assert predict1.shape == predict2.shape
        gt = gt.data.cpu().numpy().ravel()
        predict1 = predict1.detach().data.cpu().numpy().ravel()
        predict2 = predict2.detach().data.cpu().numpy().ravel()

        if considered_classes is not None:
            mask = [t in considered_classes for t in gt]
            predict1 = predict1[mask]
            predict2 = predict2[mask]

        kappa = cohen_kappa_score(y1=predict1, y2=predict2)
        self.kappa.append(kappa)

    def value(self, **kwargs):
        return torch.Tensor(self.kappa).mean()
