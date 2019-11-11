from deepclustering.arch import get_arch
from deepclustering.schedulers import GradualWarmupScheduler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

network = get_arch(arch="enet", kwargs={})
Optim = Adam(network.parameters())
scheduler = CosineAnnealingLR(Optim, T_max=1600, eta_min=1e-6)
scheduler = GradualWarmupScheduler(Optim, multiplier=100, total_epoch=700, after_scheduler=scheduler)
lrs = []
for i in range(2300):
    lrs.append(scheduler.get_lr()[0])
    scheduler.step()
import matplotlib.pyplot as plt

plt.plot(range(2300), lrs)
plt.show(block=False)
net_dict = network.state_dict()
scheduler_dict = scheduler.state_dict()
optim_dict = Optim.state_dict()
network = get_arch(arch="enet", kwargs={})
network.load_state_dict(net_dict)
Optim = Adam(network.parameters())
Optim.load_state_dict(optim_dict)
scheduler = CosineAnnealingLR(Optim, T_max=45670, eta_min=1e-6)
scheduler = GradualWarmupScheduler(Optim, multiplier=11, total_epoch=7123200, after_scheduler=scheduler)
scheduler.load_state_dict(scheduler_dict)
lrs = []
for i in range(2300, 4300):
    lrs.append(scheduler.get_lr()[0])
    scheduler.step()
import matplotlib.pyplot as plt

plt.plot(range(2300, 4300), lrs)
plt.show()
