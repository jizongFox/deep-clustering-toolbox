import torch
from generalframework.arch.deeplab import DeepLabV3
model = DeepLabV3(
    n_classes=21,
    n_blocks=[3, 4, 23, 3],
    pyramids=[6, 12, 18],
    grids=[1, 2, 4],
    output_stride=16,
)
model.freeze_bn()
model.eval()
print(list(model.named_children()))
image = torch.randn(1, 3, 513, 513)
print(model(image)[0].size())