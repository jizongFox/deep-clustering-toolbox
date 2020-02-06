# deep-clustering-toolbox
#### PyTorch Vision toolbox not only for deep-clustering
### Introduction

This repo contains the base code for a deep learning framework using `PyTorch`, to benchmark algorithms for various dataset.
The current version supports `MNIST`, `CIFAR10`, `SVHN` and `STL-10` for semisupervised and unsupervised learning.
`ACDC`, `Promise12`, `WMH` and so on are supported as segmentation counterpart.

#### Features:
>- Powerful cmd parser using `yaml` module, providing flexible input formats without predefined argparser.
>- Automatic checkpoint management adapting to various settings
>- Automatic meter recording and experimental status plotting using matplotlib and threads
>- Various build-in loss functions and help tricks and assert statements frequently used in PyTorch Framework, such as `disable_tracking_bn`, `ema`, `vat`, etc.
>- Various post-processing tools such as Viewer for Medical image segmentations, multislice_viwers for 3D dataset real-time debug
and report script for experimental summaries.
>- Extendable modules for rapid development.

#### Several projects are benefited from this scalable framework, builing top on this, including:

+ DeepClustering implemented for
>- `Invariant Information Clustering for Unsupervised Image Classification and Segmentation`,
>- `Learning Discrete Representations via Information Maximizing Self-Augmented Training`,
>- [`Information based Deep Clustering: An experimental study`](https://github.com/jizongFox/DeepClusteringProject)
+ SemiSupervised classification for
>- `Semi-Supervised Learning by Augmented Distribution Alignment`,
>- `Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning`,
>- `Temporal Ensembling for Semi-Supervised Learning`,
>- `Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results`
+ SemiSupervised Segmentation for
>- `Adversarial Learning for Semi-Supervised Semantic Segmentation`,
>- `Semi-Supervised and Task-Driven Data Augmentation`,
>- [`Deep Co-Training for Semi-Supervised Image Segmentation`](https://arxiv.org/abs/1903.11233)
+ Discretely-constrained CNN for
>- [`Discretely-constrained deep network for weakly-supervised segmentation`](https://github.com/jizongFox/Discretly-constrained-CNN/).


They are examples how to develop research framework with the assistance of our proposed `deep-clustering-toolbox`.
___
### Playground

Several papers have been implemented based on this framework. I store them in the `playground` folder. The papers include:

>- [`Auto-Encoding Variational Bayes`](https://arxiv.org/abs/1312.6114)
>- [`mixup: BEYOND EMPIRICAL RISK MINIMIZATION`](https://arxiv.org/pdf/1710.09412.pdf)
>- [`MINE: Mutual Information Neural Estimation`](https://arxiv.org/abs/1801.04062)
>- [`Averaging Weights Leads to Wider Optima and Better Generalization`](https://arxiv.org/pdf/1803.05407.pdf)
>- [`THERE ARE MANY CONSISTENT EXPLANATIONS OF UNLABELED DATA: WHY YOU SHOULD AVERAGE`](https://arxiv.org/pdf/1806.05594.pdf)
>- [`Prior-aware Neural Network for Partially-Supervised Multi-Organ Segmentation`](https://arxiv.org/abs/1904.06346)


---
### Installation
```bash
git clone https://github.com/jizongFox/deep-clustering-toolbox.git
cd deep-clustering-toolbox  
python setup install # for those who do not want to make changes immediately.
# or
python setup develop # for those who want to modify the code and make the impact immediate.

```
Or very simply
```bash
pip install deepclustering
```
### Citation
If you feel useful for your project, please consider citing this work.
```latex
@article{peng2019deep,
  title={Deep Co-Training for Semi-Supervised Image Segmentation},
  author={Peng, Jizong and Estradab, Guillermo and Pedersoli, Marco and Desrosiers, Christian},
  journal={arXiv preprint arXiv:1903.11233},
  year={2019}
}
```
