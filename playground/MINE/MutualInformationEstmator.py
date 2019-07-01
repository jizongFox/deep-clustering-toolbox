"""
This function is to calculate the MI.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm


def ma(a, window_size=100):
    return [np.mean(a[i : i + window_size]) for i in range(0, len(a) - window_size)]


class Mine(nn.Module):
    def __init__(self, input_size=2, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self._init_weights()

    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        # the output is the logit number!!!!
        return output

    def _init_weights(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, std=0.02)
        nn.init.constant_(self.fc3.bias, 0)


class MI_Estimator(object):
    def __init__(self, model: nn.Module, moving_average_et=1, ma_ratio=0.01) -> None:
        super().__init__()
        self.model = model
        self.moving_average_et = moving_average_et
        self.moving_average_ratio = ma_ratio

    def __call__(self, joint_dis, margin_dis):
        t = self.model(joint_dis)
        exp_m = torch.exp(self.model(margin_dis))
        mi_lowbound = torch.mean(t) - torch.log(torch.mean(exp_m))
        self.moving_average_et = (
            1 - self.moving_average_ratio
        ) * self.moving_average_et + self.moving_average_ratio * torch.mean(exp_m)

        loss = -(
            torch.mean(t)
            - (1 / self.moving_average_et.mean()).detach() * torch.mean(exp_m)
        )  # unbiased estimation of the MI
        return mi_lowbound, loss


def mutual_information(joint, marginal, mine_net, ma_et=1.0, ma_rate=0.01):
    """
    interface for the mutual information.
    :param joint: Joint distribution of the two variables
    :param marginal: Marginal distribution of the two variables
    :param mine_net: the network to simulate the lowerbound
    :return: MI and loss
    """
    t = mine_net(joint)
    exp_marginal = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(
        torch.mean(exp_marginal)
    )  # this is the MI to record
    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(exp_marginal)
    # unbiasing use moving average
    loss = -(torch.mean(t) - (1 / ma_et.mean()).detach() * torch.mean(exp_marginal))
    return mi_lb, loss, ma_et


def sample_batch(data, batch_size=100, sample_mode="joint"):
    if sample_mode == "joint":
        index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = data[index]
    else:
        joint_index = np.random.choice(
            range(data.shape[0]), size=batch_size, replace=False
        )
        marginal_index = np.random.choice(
            range(data.shape[0]), size=batch_size, replace=False
        )
        batch = np.concatenate(
            [
                data[joint_index][:, 0].reshape(-1, 1),
                data[marginal_index][:, 1].reshape(-1, 1),
            ],
            axis=1,
        )
    return batch


if __name__ == "__main__":
    # prepare data
    x = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=3000)
    # the mutual information should be 0
    y = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.9], [0.9, 1]], size=3000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = y
    batch_size = 1000
    mine_net = Mine().cuda()
    mine_net_optim = optim.Adam(mine_net.parameters(), lr=1e-3)

    # data is x or y
    result = list()
    ma_et = 1.0
    iter_num = tqdm(range(50000))
    for i in iter_num:
        joint, margin = (
            sample_batch(data, batch_size=batch_size),
            sample_batch(data, batch_size=batch_size, sample_mode="marginal"),
        )
        joint, margin = (
            torch.Tensor(joint).float().to(device),
            torch.Tensor(margin).float().to(device),
        )
        mi_lb, loss, ma_et = mutual_information(
            joint, marginal=margin, mine_net=mine_net, ma_et=ma_et
        )
        mine_net_optim.zero_grad()
        (loss).backward()
        mine_net_optim.step()
        result.append(mi_lb.detach().cpu().numpy())
        iter_num.set_postfix({"MI": result[-1]})
    plt.plot(range(len(ma(result))), ma(result))
    plt.show()
