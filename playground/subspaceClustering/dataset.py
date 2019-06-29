import numpy as np
from torch.utils.data import Dataset


class TensorDataset(Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors), index

    def __len__(self):
        return self.tensors[0].size(0)


def create_gaussian_norm_dataset(
    num_cluster: int, num_exp: int, num_feature: int, var: float, shuffle=True
) -> np.ndarray:
    """
    :param num_cluster:  Number of clusters
    :param num_exp: Number of examples per cluster
    :param num_feature: Number of features per example
    :param var: Variance of each cluster
    :return: numpy dataset
    """
    dataset = np.zeros((num_cluster * num_exp, num_feature))
    centers: np.ndarray = np.random.randn(
        num_cluster, num_feature
    )  # :shape  num_cluster * num_feature
    batch_num = 0
    for c in centers:
        for _ in range(num_exp):
            dataset[batch_num] = c + var * np.random.randn(*c.shape)
            batch_num += 1
    assert batch_num == num_cluster * num_exp
    if shuffle:
        np.random.shuffle(dataset)  # inplace operation
    return dataset


def merge(*args, dim=0):
    return torch.cat(args, dim=dim)
