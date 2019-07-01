import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def fix_seed(seed: int) -> None:
    assert isinstance(seed, int)
    np.random.seed(seed)


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


fix_seed(1)

dataset = create_gaussian_norm_dataset(10, 50, 2, 0.1, shuffle=False)

# ADMM loop
mu = 1
max_epoch = 100
lamda = 0.01

S = np.zeros([dataset.shape[0]] * 2)
U = np.zeros_like(S)

# todo: what does X2 mean here? to form a symmetric one?
X2 = dataset.dot(dataset.transpose())
assert X2.shape == tuple([dataset.shape[0]] * 2)
# todo: what does the T here mean.
T = np.linalg.inv(X2 + mu * np.eye(X2.shape[0]))
plt.ion()
for epoch in tqdm(range(max_epoch)):
    Z = T.dot(X2 + mu * (S + U))
    S = np.maximum(Z - U - (lamda / mu), 0)
    np.fill_diagonal(S, 0)
    U = U + (S - Z)

    res = np.linalg.norm(S - Z, ord="fro")
    err = np.linalg.norm(dataset - S.dot(dataset), ord="fro") / np.linalg.norm(
        dataset, ord="fro"
    )

    plt.imshow(S)
    plt.show()
    plt.pause(0.1)
plt.ioff()
