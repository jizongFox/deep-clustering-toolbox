import matplotlib
import numpy as np
from tensorboardX import SummaryWriter as _SummaryWriter
matplotlib.use('agg')
import matplotlib.pyplot as plt


class SummaryWriter(_SummaryWriter):
    def __init__(self, log_dir=None, comment='', **kwargs):
        assert log_dir is not None, f'log_dir should be provided, given {log_dir}.'
        log_dir = str(log_dir) + '/tensorboard'
        super().__init__(log_dir, comment, **kwargs)


class DrawCSV(object):
    def __init__(self, columns_to_draw=None, save_dir=None, save_name='plot.png', figsize=[10, 15]) -> None:
        super().__init__()
        if columns_to_draw is not None and not isinstance(columns_to_draw, list):
            columns_to_draw = [columns_to_draw]
        self.columns_to_draw = columns_to_draw
        self.save_name = save_name
        self.save_dir = save_dir
        self.figsize = tuple(figsize)

    def draw(self, dataframe, together=False):
        if together:
            fig = plt.figure(figsize=self.figsize)
            for k in self.columns_to_draw:
                plt.plot(dataframe[k], label=k)
            plt.legend()
            plt.grid()
            plt.savefig(str(self.save_dir) + f'/{self.save_name}')
        else:
            fig, axs = plt.subplots(nrows=self.columns_to_draw.__len__(), sharex=True, figsize=self.figsize)
            if not isinstance(axs, np.ndarray):
                axs = np.array([axs])
            for i, k in enumerate(self.columns_to_draw):
                _ax = axs[i]
                _ax.plot(dataframe[k], label=k)
                _ax.legend()
                _ax.grid()
                _ax.set_title(f"{k} with max:{dataframe[k].max():.3f}, min:{dataframe[k].min():.3f}")
            plt.savefig(str(self.save_dir) + f'/{self.save_name}')
        plt.close(fig)
