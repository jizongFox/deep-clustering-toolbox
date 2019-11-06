# this is the toy example to optimize the primal-dual gradient descent.
# toy example from sklearn
import matplotlib

matplotlib.use('qt5agg')
import torch

from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot


# generate 2d classification dataset
X, y = make_blobs(n_samples=5000, centers=3, n_features=20, random_state=0)
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.1, random_state=0)



X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()

