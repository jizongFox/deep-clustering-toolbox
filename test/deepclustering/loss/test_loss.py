import unittest

import torch

from deepclustering.loss import loss


class Test_CrossEntropyLoss2D(unittest.TestCase):
    def setUp(self):
        self.weight = torch.Tensor([1, 2, 3, 4])
        self.loss = loss.CrossEntropyLoss2d
        self.predict = torch.randn(10, 4, 224, 224)
        b, c, h, w = self.predict.shape
        self.label = torch.randint(0, 4, size=(b, h, w))

    def test_weight(self):
        self.criterion = self.loss(weight=self.weight)
        loss = self.criterion(self.predict, self.label)
        # assert loss ==1

    def test_cuda(self):
        for arg in self.__dict__:
            if isinstance(arg, torch.Tensor):
                arg = arg.to('cpu')

        self.weight = self.weight.to('cuda')
        self.predict = self.predict.cuda()
        self.label = self.label.cuda()

        self.test_weight()
