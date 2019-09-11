# deep-clustering-toolbox
### Introduction

This repo contains the base code for a deep learning framework, used to benchmark algorithms for various dataset. 
The current version supports `MNIST`, `CIFAR10`, `SVHN` and `STL-10` for semi supervised and un supervised learning. 
`ACDC`, `Promise12` and `WMH` are supported as segmentation counterpart.

Several projects are rooted from this repo, including: 

+ DeepClustering implemented for 
>- `Invariant Information Clustering for Unsupervised Image Classification and Segmentation`, 
>- `Learning Discrete Representations via Information Maximizing Self-Augmented Training`
+ SemiSupervised Clustering for 
>- `Semi-Supervised Learning by Augmented Distribution Alignment`, 
>- `Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning`, 
>- `Temporal Ensembling for Semi-Supervised Learning`,
>- `Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results`
+ SemiSupervised Segmentation for 
>- `Adversarial Learning for Semi-Supervised Semantic Segmentation`, 
>- `Semi-Supervised and Task-Driven Data Augmentation`, etc.
+ Discretely-constrained CNN for
>- [`Discretely-constrained deep network for weakly-supervised segmentation`](https://github.com/jizongFox/Discretly-constrained-CNN/).
___
### Playground

Several papers have been implemented based on this framework. I store them in the `playground` folder. The papers include:

>- [`Auto-Encoding Variational Bayes`](https://arxiv.org/abs/1312.6114)
>- [`mixup: BEYOND EMPIRICAL RISK MINIMIZATION`](https://arxiv.org/pdf/1710.09412.pdf)
>- [`MINE: Mutual Information Neural Estimation`](https://arxiv.org/abs/1801.04062)
>- [`Averaging Weights Leads to Wider Optima and Better Generalization`](https://arxiv.org/pdf/1803.05407.pdf)
>- [`THERE ARE MANY CONSISTENT EXPLANATIONS OF UNLABELED DATA: WHY YOU SHOULD AVERAGE`](https://arxiv.org/pdf/1806.05594.pdf)



---
### Installation
```bash
git clone current repo  
cd deep-clustering-toolbox  
python setup install / develop.
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



