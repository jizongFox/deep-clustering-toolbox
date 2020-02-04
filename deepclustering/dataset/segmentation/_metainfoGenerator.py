from typing import List

from torch import Tensor

from deepclustering.utils import class2one_hot


class classSizeCalulator:
    def __init__(self, C: int, foldernames: List[str]) -> None:
        super().__init__()
        self.C = C
        self.foldernames = foldernames

    def __call__(self, Seg: Tensor):
        assert Seg.shape.__len__() == 3
        onehotSeg = class2one_hot(Seg, self.C)
        assert onehotSeg.shape.__len__() == 4
        classSize = onehotSeg.sum([0, 2, 3])
        assert classSize.shape[0] == self.C
        return classSize


class getImage_GT:
    def __init__(self, foldernames: List[str]) -> None:
        super().__init__()
        self.foldernames = foldernames

    def __call__(self, x):
        return x
