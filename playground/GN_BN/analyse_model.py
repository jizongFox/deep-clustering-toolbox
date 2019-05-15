import torch
from pathlib2 import Path
from resnet import resnet18, NormalizationLayer
from torch import nn


def analyze_alpha(model_cls, checkpoint_path, epoch, save_dir):
    assert isinstance(save_dir, Path)
    save_dir.mkdir(exist_ok=True, parents=True)
    state_dict = torch.load(str(checkpoint_path),
                            map_location=torch.device('cpu'))
    resnet: nn.Module = model_cls(10)
    resnet.load_state_dict(state_dict['model_state_dict']['net_state_dict'])
    print(f'checkpoint loaded with acc: {state_dict["best_score"]:.4f} at {state_dict["epoch"]} epoch.')

    model = resnet
    l = {k: module for k, module in model.named_modules() if type(module) != nn.Sequential}
    alpha_dict = {}
    for k, m in l.items():
        if isinstance(m, NormalizationLayer):
            alpha_dict[k] = torch.sigmoid(m.alpha).item()
    import matplotlib.pyplot as plt

    plt.plot(range(len(alpha_dict.keys())), alpha_dict.values())
    plt.xticks(range(len(alpha_dict.keys())), list(alpha_dict.keys()), rotation='vertical')
    plt.grid()
    plt.show()
    plt.savefig(str(Path(save_dir) / f'alpha_{epoch}.png'))
    plt.close('all')
