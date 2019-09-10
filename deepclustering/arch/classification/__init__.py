"""
This folder is taken from IIC paper: https://github.com/xu-ji/IIC
"""
from deepclustering.arch.classification.IIC.baselines import *
from deepclustering.arch.classification.IIC.net5g import *
from deepclustering.arch.classification.IIC.net5g_multi_head import *
from deepclustering.arch.classification.IIC.net5g_two_head import *
from deepclustering.arch.classification.IIC.net6c import *
from deepclustering.arch.classification.IIC.net6c_two_head import *
from deepclustering.arch.classification.IMSAT.imsat import *
from deepclustering.arch.classification.dummy import *
from deepclustering.arch.classification.vat_network import *
from deepclustering.arch.classification.preresnet import *

"""
5G net is based on resnet34 while 6C net is based on VGG16.
"""
