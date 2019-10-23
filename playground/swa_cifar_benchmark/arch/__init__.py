from deepclustering.arch import _register_arch

from .densenet import *
from .dpn import *
from .efficientnet import *
from .googlenet import *
from .large_conv import LargeConvNet
from .lenet import *
from .mobilenet import *
from .mobilenetv2 import *
from .pnasnet import *
from .preact_resnet import *
from .resnet import *
from .resnext import *
from .senet import *
from .shufflenet import *
from .shufflenetv2 import *
from .vgg import *
from .wideresnet import WideResNet

# register architecture
_register_arch("vgg", VGG)
_register_arch("resnet18", ResNet18)
_register_arch("dpn92", DPN92)
_register_arch("mobilenetv2", MobileNetV2)
_register_arch("wideresnet", WideResNet)
_register_arch("largeconvnet", LargeConvNet)
