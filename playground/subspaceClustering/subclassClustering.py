from typing import List

import torch
from deepclustering.method import _Method
from deepclustering.model import Model
from torch import Tensor


class SubSpaceClusteringMethod(_Method):
    def __init__(
        self,
        model: Model,
        lamda: float = 0.1,
        lr: float = 0.0001,
        num_samples: int = 100,
        device: torch.device = torch.device("cuda"),
        *args,
        **kwargs,
    ):
        super().__init__(model, *args, **kwargs)
        assert isinstance(
            device, torch.device
        ), f"device should be torch.device, given {device}."
        self.lr = float(lr)
        self.lamda = float(lamda)
        self.device = device
        self.adj_matrix = torch.randn(
            (num_samples, num_samples), dtype=torch.float32
        ).to(self.device)
        self._diagnoal_remove(self.adj_matrix)
        # self.adj_matrix = torch.eye(num_samples).to(self.device) #+ 0.1*torch.randn((num_samples,num_samples)).to(device)*torch.eye(num_samples).to(self.device)
        print()

    def _diagnoal_remove(self, matrix):
        assert (
            matrix.shape.__len__() == 2 and matrix.shape[0] == matrix.shape[1]
        ), f"check the matrix dimension, given {matrix.shape}"
        for i in range(len(matrix)):
            matrix[i, i] = 0
        assert self.check_diagnal_zero(matrix), f"matrix diag remove failed."

    @staticmethod
    def check_diagnal_zero(matrix: Tensor) -> bool:
        return torch.allclose(matrix.diag(), torch.zeros_like(matrix.diag()))

    def set_input(self, imgs: Tensor, index: List[int], *args, **kwargs):
        super().set_input(*args, **kwargs)
        assert imgs.shape[0] == len(index), (
            f"imgs and index lengths should be the same, given len(imgs)="
            f"{len(imgs)}, len(index)={len(index)}."
        )
        self.imgs = imgs
        # self.pred, self._representation = self.model(self.imgs)
        self._representation = self.imgs.view(self.imgs.shape[0], -1)
        self.index = index

        assert self._representation.shape[0] == self.index.shape[0]
        self.current_adj_matrix: Tensor = self.adj_matrix[index][:, index]
        assert self.current_adj_matrix.shape == torch.Size([len(index), len(index)])
        # choose the minibatch of adj_matrix

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._update_dictionary()
        # self._gradient_descent()

    def _gradient_descent(self):
        current_adj_matrix = self.current_adj_matrix.clone()
        self._diagnoal_remove(current_adj_matrix)
        _reconstr_loss = (
            (self._representation - torch.mm(current_adj_matrix, self._representation))
            .norm(p=2, dim=1)
            .mean()
        )
        self.model.zero_grad()
        _reconstr_loss.backward()
        self.model.step()
        # print(_reconstr_loss)

    def _update_dictionary(self):
        assert self.check_diagnal_zero(self.current_adj_matrix)
        X2 = self._representation.mm(self._representation.t()).detach()
        I = torch.eye(len(self.current_adj_matrix)).to(self.device)
        for _ in range(1000):
            current_adj_matrix_hat = self.current_adj_matrix - self.lr * X2.mm(
                self.current_adj_matrix - I
            )
            current_adj_sign = current_adj_matrix_hat.sign()
            new_current_adj = (
                torch.max(
                    current_adj_matrix_hat.__abs__() - self.lr * self.lamda,
                    torch.zeros_like(current_adj_matrix_hat),
                )
                * current_adj_sign
            )
            self._diagnoal_remove(new_current_adj)
            self.current_adj_matrix = new_current_adj
        # update the whole matrix
        for i, c in enumerate(self.index):
            self.adj_matrix[c, self.index] = new_current_adj[:, i]  # new_current_adj
        # self.adj_matrix.scatter((self.index, self.index), -1000)


class SubSpaceClusteringMethod2(SubSpaceClusteringMethod):
    def __init__(
        self,
        model: Model,
        lamda: float = 0.1,
        lr: float = 0.005,
        num_samples: int = 100,
        device: torch.device = torch.device("cuda"),
        *args,
        **kwargs,
    ):
        super().__init__(model, lamda, lr, num_samples, device, *args, **kwargs)

    def _update_dictionary(self):
        # reconstruction:
        current_adj_matrix = self.current_adj_matrix.clone()
        for _ in range(1000):
            self._diagnoal_remove(current_adj_matrix)
            current_adj_matrix.requires_grad = True
            _reconstr_loss = (
                (
                    self._representation
                    - torch.mm(current_adj_matrix, self._representation)
                )
                .norm(p=2, dim=1)
                .mean()
            )
            _sparsity_loss = current_adj_matrix.norm(p=1, dim=0).mean()
            _loss = _reconstr_loss + _sparsity_loss
            _loss.backward()
            # print(f"sparsity:{_sparsity_loss}, reconstruction:{_reconstr_loss}")
            new_current_adj_matrix = (
                current_adj_matrix - self.lamda * current_adj_matrix.grad
            )
            new_current_adj_matrix = new_current_adj_matrix.detach()
            current_adj_matrix.grad.zero_()

            # new_current_adj_matrix[new_current_adj_matrix.__abs__()<=0.0001]=0 #* torch.eye(len(self.index)).to(self.device)
            self._diagnoal_remove(new_current_adj_matrix)
            current_adj_matrix = new_current_adj_matrix
        for i, c in enumerate(self.index):
            self.adj_matrix[c, self.index] = new_current_adj_matrix[
                i
            ]  # new_current_adj
        print(
            f"reconstruction:{_reconstr_loss}, sparsity:{_sparsity_loss}, current_adj_max:{new_current_adj_matrix.diag().max()}, min:{new_current_adj_matrix.diag().min()}"
        )

    def update(self):
        self._update_dictionary()
