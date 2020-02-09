import argparse
from pprint import pprint
from typing import List, Union

import pandas as pd


class DrawCSV2(object):
    def __init__(
        self,
        columns_to_draw=None,
        save_dir=None,
        save_name="plot.png",
        csv_name="wholeMeter.csv",
        figsize=[10, 15],
    ) -> None:
        super().__init__()
        if columns_to_draw is not None and not isinstance(columns_to_draw, list):
            columns_to_draw = [columns_to_draw]
        self.columns_to_draw: List[Union[str, List[str]]] = columns_to_draw
        self.save_name = save_name
        self.save_dir = save_dir
        self.figsize = tuple(figsize)
        self.csv_name = csv_name

    # @threaded(daemon=False)
    def draw(self, dataframe):
        import matplotlib
        import numpy as np

        matplotlib.use("agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(
            nrows=self.columns_to_draw.__len__(), sharex=True, figsize=self.figsize
        )
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        for i, (_ax, k) in enumerate(zip(axs, self.columns_to_draw)):
            if isinstance(k, str):
                self._draw_single(_ax, dataframe, k)
            else:
                self._draw_mulitple(_ax, dataframe, k)

        plt.savefig(str(self.save_dir) + f"/{self.save_name}")
        plt.close(fig)

    def _draw_single(self, ax, data_frame: pd.DataFrame, column_name: str):
        ax.plot(
            pd.Series(data_frame[column_name]).fillna(limit=5, method="ffill"),
            label=column_name,
        )
        ax.legend()
        ax.grid()
        ax.set_title(
            f"{column_name} with max:{data_frame[column_name].max():.3f}, "
            f"min:{data_frame[column_name].min():.3f}"
        )

    def _draw_mulitple(self, ax, data_frame: pd.DataFrame, column_names: List[str]):
        for k in column_names:
            ax.plot(pd.Series(data_frame[k]).fillna(limit=5, method="ffill"), label=k)
            # ax.plot(data_frame[k], label=k)
        ax.legend()
        ax.grid()
        title = (
            lambda k: f"{k} with max:{data_frame[k].max():.3f}, min:{data_frame[k].min():.3f}"
        )
        ax.set_title("\n".join([title(k) for k in column_names]))


def arg_parser() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="parser to get parameters to draw csv using matplotlib."
    )
    parser.add_argument("--csv_path", type=str, required=True, help="csv path to draw.")
    parser.add_argument(
        "--columns_to_draw",
        required=True,
        type=str,
        nargs="+",
        metavar="columnA, columnB, ...",
        help="Columns to draw",
    )
    parser.add_argument(
        "--save_dir", required=True, type=str, help="save_dir for the plot."
    )
    parser.add_argument(
        "--save_name", type=str, default="plot.png", help="Name of the plot."
    )
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=[10, 15],
        metavar="A B",
        help="figure size, default [10, 15]",
    )
    parser.add_argument(
        "--overlap",
        action="store_true",
        default=False,
        help="Whether overlap plots by columns.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="show args information", default=False
    )
    args = parser.parse_args()
    if args.verbose:
        print("Arguments:")
        pprint(args)
    return args
