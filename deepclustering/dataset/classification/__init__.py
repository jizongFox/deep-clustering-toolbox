from .cifar import CIFAR10
from .cifar_helper import (
    Cifar10ClusteringDatasetInterface,
    default_cifar10_img_transform,
    Cifar10SemiSupervisedDatasetInterface,
)
from .mnist import MNIST
from .mnist_helper import (
    MNISTClusteringDatasetInterface,
    default_mnist_img_transform,
    MNISTSemiSupervisedDatasetInterface,
)
from .stl10 import STL10
from .stl10_helper import STL10DatasetInterface, default_stl10_img_transform
from .svhn import SVHN
from .svhn_helper import (
    SVHNSemiSupervisedDatasetInterface,
    SVHNClusteringDatasetInterface,
    svhn_naive_transform,
    svhn_strong_transform,
)
