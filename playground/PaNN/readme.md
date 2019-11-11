# Prior-aware Netrual Network (PaNN) and its optimization
---

#### Verification of Optimization proposed in section `3.2` and `3.3` of the paper [`Prior-aware Neural Network for Partially-Supervised Multi-Organ Segmentation`](https://arxiv.org/abs/1904.06346)

<img src="https://latex.codecogs.com/svg.latex?\Large&space;KL_{\text{marginal}}(q|\overline{p})=-qlog(\overline{p})+qlog(q)" />  


where `q` is a predefined distribution which is fixed and <img src="https://latex.codecogs.com/svg.latex?\Large&space;\overline{p}" /> is the marginal distribution of all pixel predictions

There might be some mistakes of the original paper, `Equ. 2` has redundent components, meaning an increased learning rate for optimization.

The paper claims that the optimization of  <img src="https://latex.codecogs.com/svg.latex?\Large&space;\log(\overline{p})" /> = `log(mean(all_pixel_predictions))` may be problematic thus biased. This kined of optimization is in fact widely used in `Mutual information maximization` or `conditional entropy minimization`.
See my [paper](https://arxiv.org/abs/1910.01665) for a reference. 

The Interesting solution in the paper would be to use the `primal-dual gradient` method. This folder reproduce this kind of optimization for semi-supervised learning with ACDC dataset to understand the advantage of the proposed method.

---
#### using directly entropy method:
<img src="./.figures/baseline_entropy.png" alt="using directly entropy method" style="zoom:10%;" />
#### using Primal-dual method:
<img src="./.figures/Primal-dual_Entropy.png" alt="using PD method" style="zoom:10%;" />
We can see that PD can significantly improve the val acc for `sup_loss + reg_loss` where `reg_loss` is kl(q_fix|mean(p))
----
#### We added a conditional entropy term.
<img src="./.figures/entropy_centropy.png" alt="using directly entropy method" style="zoom:10%;" />
#### using Primal-dual method:
<img src="./.figures/Primal-dual_centropy.png" alt="using PD method" style="zoom:10%;" />
We can see the Entropy -CEntropy gives more stability and convergence speed.



---
### We tested different optimizers in this segmentation task, including `Adam`, `RAdam`, and `AdaBound`.

training loss:
<img src="./.figures/train_loss.png" alt="training curve for segmentation using CE" style="zoom:10%;" />
validation dice:
<img src="./.figures/3D_dice.png" alt="validation_dice" style="zoom:10%;" />