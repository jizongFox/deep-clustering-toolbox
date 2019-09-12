# Averaging Weights Leads to Wider Optima and Better Generalization

This folder reproduces the algorithm proposed in the paper [`Averaging Weights Leads to Wider Optima and Better Generalization`](https://arxiv.org/abs/1803.05407), based on the code released [here](https://github.com/timgaripov/swa).

***
This folder also reimplemented the benchmark cifar10 based on `https://github.com/kuangliu/pytorch-cifar/tree/master/models`

```
| network   | reported | implemented| SWA |
| resnet-18 |  93.02%  | 95.55% | 95.1%
|MobileNetV2|  94.43%  | 91.40% |
```

# todo
***
Compared with original code to see the implementation difference leading to failure of a better accuracy compared with the SGD.