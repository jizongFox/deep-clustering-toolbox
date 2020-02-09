from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt

from deepclustering.decorator import threaded
from deepclustering.meters import MeterInterface

# here we should pass a callback function to specific a different plot style.
from ._dataframedrawercallback import (
    multipleline_plot as default_plot_callback,
    plot_callback,
    _repositioning_axes,
)


class DataFrameDrawer:
    def __init__(
        self,
        meterinterface: MeterInterface,
        save_dir: Union[str, Path],
        save_name: str = "dataframeDrawer.png",
    ) -> None:
        self._meterinterface = meterinterface
        self._meter_names = self._meterinterface.meter_names
        self._callback_dict = {}
        self._save_dir = save_dir if isinstance(save_dir, Path) else Path(save_dir)
        self._save_dir.mkdir(exist_ok=True, parents=True)
        self._save_name = save_name

    def _draw(self):
        fig = plt.figure(figsize=(5, 2))
        self._meter_names = self._meterinterface.meter_names
        summary_dataframes = self._meterinterface.history_summary()
        assert len(self._meter_names) == len(summary_dataframes)
        for i, m_name in enumerate(self._meter_names):
            plot_names = self._meterinterface.individual_meters[m_name].get_plot_names()
            self._callback_dict.get(m_name, default_plot_callback())(
                fig, summary_dataframes[m_name][plot_names], title=m_name
            )
        _repositioning_axes(fig)
        return fig

    def _save_fig(self, fig):
        fig.set_size_inches(5, 2 * len(fig.axes), forward=True)
        fig.savefig(self._save_dir / self._save_name, dpi=600)
        plt.close("all")

    @threaded(name="drawer", daemon=False)
    def __call__(self):
        fig = self._draw()
        self._save_fig(fig)

    def set_callback(self, meter_name, plt_callback):
        assert meter_name in self._meterinterface.meter_names, meter_name
        assert isinstance(plt_callback, plot_callback), plt_callback
        self._callback_dict[meter_name] = plt_callback
