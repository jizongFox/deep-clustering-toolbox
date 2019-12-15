# using singledispath to deal with image augmentation.

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def inverse_transform_matrix(AffineMatrix: torch):
    def _inverse_transform_matrix(AffineMatrix: torch):
        k, l = AffineMatrix.shape
        assert k == 2, l == 3
        fullMatrix = torch.cat(
            [
                AffineMatrix,
                torch.from_numpy(
                    np.array([0, 0, 1]).reshape(1, 3).astype(np.float32)
                ).to(AffineMatrix.device),
            ],
            dim=0,
        )
        InverseMatrix = torch.inverse(fullMatrix)[:2, :]
        return InverseMatrix

    bn, k, l = AffineMatrix.shape
    assert k == 2, l == 3
    InverseAffineMatrix = []
    for matrix in AffineMatrix:
        InverseAffineMatrix.append(_inverse_transform_matrix(matrix))
    return torch.stack(InverseAffineMatrix, dim=0)


class AffineTensorTransform(object):
    def __init__(
        self,
        min_rot: float = 0,
        max_rot: float = 180,
        min_shear: float = 0.0,
        max_shear: float = 0.5,
        min_scale: float = 0.7,
        max_scale: float = 1.3,
        mode="bilinear",
    ):
        self.min_rot = min_rot
        self.max_rot = max_rot
        self.min_shear = min_shear
        self.max_shear = max_shear
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.mode = mode

    def get_random_affinematrix(self):
        a = np.radians(np.random.rand() * (self.max_rot - self.min_rot) + self.min_rot)
        shear = np.radians(
            np.random.rand() * (self.max_shear - self.min_shear) + self.min_shear
        )
        scale = np.random.rand() * (self.max_scale - self.min_scale) + self.min_scale

        RandomAffineMatrix = np.array(
            [
                [np.cos(a) * scale, -np.sin(a + shear) * scale, 0.0],
                [np.sin(a) * scale, np.cos(a + shear) * scale, 0.0],
            ],
            dtype=np.float32,
        )  # 3x3
        return torch.from_numpy(RandomAffineMatrix).float()

    def _perform_affine_transform(self, single_img: Tensor, affinematrix: Tensor):
        assert len(single_img.shape) == 4
        assert affinematrix.shape[1] == 2, affinematrix.shape[2] == 3
        affinematrix = affinematrix.to(single_img.device)
        grid = F.affine_grid(
            affinematrix, single_img.shape
        )  # output should be same size
        data_tf = F.grid_sample(
            single_img, grid, mode=self.mode, padding_mode="zeros"
        )  # this can ONLY do bilinear
        return data_tf

    def __call__(
        self, img: Tensor, AffineMatrix: Tensor = None, independent=True, inverse=False
    ):
        assert img.shape.__len__() in (4, 3, 2)
        _img_shape = img.shape
        img = img.unsqueeze(0) if len(img.shape) == 2 else img
        img = img.unsqueeze(0) if len(img.shape) == 3 else img
        assert img.shape.__len__() == 4
        bn = img.shape[0]

        if AffineMatrix is None:
            AffineMatrix = self.get_random_affinematrix()
            if independent:
                AffineMatrix = torch.stack([self.get_random_affinematrix()] * bn, dim=0)
            else:
                AffineMatrix = torch.stack([AffineMatrix] * bn, dim=0)
        else:
            AffineMatrix.shape[0] == bn

        if inverse:
            AffineMatrix = inverse_transform_matrix(AffineMatrix)
        img_tf = self._perform_affine_transform(img, AffineMatrix)
        img_tf = img_tf.view(*_img_shape)
        return img_tf, AffineMatrix
