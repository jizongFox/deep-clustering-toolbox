# callback function for plot, given axe, dataframe, and plot_name
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox


def _num_axes(fig):
    return len(fig.axes)


def _repositioning_axes(fig):
    n = _num_axes(fig)
    if n > 0:
        heights = np.linspace(0.05, 0.95, n + 1)
        for i, ax in enumerate(fig.axes):
            ax.set_position(
                Bbox([[0.10, heights[i] + 0.03], [0.99, heights[i + 1] - 0.03]]),
                which="both",
            )


def _create_new_axe(fig):
    n = _num_axes(fig)
    ax = fig.add_subplot(n + 1, 1, n + 1)
    return ax


class plot_callback:
    def __call__(
        self, fig: Figure, data_frame: pd.DataFrame, title: str = None, *args, **kwargs
    ):
        pass


class singleline_plot(plot_callback):
    def __call__(
        self, fig: Figure, data_frame: pd.DataFrame, title: str = None, *args, **kwargs
    ):

        column_names = list(data_frame.columns)
        for i, k in enumerate(column_names):
            ax = _create_new_axe(fig)
            ax.plot(pd.Series(data_frame[k]).fillna(limit=5, method="ffill"), label=k)
            ax.legend()
            ax.grid()
            if title:
                _title = (
                    lambda k: f"{title}: {k} with max:{data_frame[k].max():.3f}, min:{data_frame[k].min():.3f}"
                )
                ax.set_title("\n".join([_title(k)]), fontsize=8)


class multipleline_plot(plot_callback):
    def __call__(
        self, fig: Figure, data_frame: pd.DataFrame, title: str = None, *args, **kwargs
    ):
        column_names = list(data_frame.columns)
        ax = _create_new_axe(fig)
        for k in column_names:
            ax.plot(pd.Series(data_frame[k]).fillna(limit=5, method="ffill"), label=k)
        ax.legend()
        ax.grid()
        if title:
            _title = (
                lambda k: f"{title}: {k} with max:{data_frame[k].max():.3f}, min:{data_frame[k].min():.3f}"
            )
            ax.set_title("\n".join([_title(k) for k in column_names]), fontsize=8)
