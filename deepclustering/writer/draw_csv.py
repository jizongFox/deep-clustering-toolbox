import argparse
import os
import subprocess
from pathlib import Path
from pprint import pprint

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('agg')
import matplotlib.pyplot as plt
from deepclustering import PROJECT_PATH


class DrawCSV(object):
    def __init__(
            self,
            columns_to_draw=None,
            save_dir=None,
            save_name='plot.png',
            csv_name='wholeMeter.csv',
            figsize=[10, 15]
    ) -> None:
        super().__init__()
        if columns_to_draw is not None and not isinstance(columns_to_draw, list):
            columns_to_draw = [columns_to_draw]
        self.columns_to_draw = columns_to_draw
        self.save_name = save_name
        self.save_dir = save_dir
        self.figsize = tuple(figsize)
        self.csv_name = csv_name

    def call_draw(self):
        _csv_path = os.path.join(str(self.save_dir), self.csv_name)
        _columns_to_draw = " ".join(self.columns_to_draw)
        _save_dir = str(self.save_dir)
        cmd = f"python  {PROJECT_PATH}/deepclustering/writer/draw_csv.py  " \
            f"--csv_path={_csv_path} " \
            f"--save_dir={_save_dir} " \
            f"--columns_to_draw {_columns_to_draw} &"
        subprocess.call(cmd, shell=True)

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
                if len(dataframe[k]) == 0:
                    continue
                _ax = axs[i]
                _ax.plot(dataframe[k], label=k)
                _ax.legend()
                _ax.grid()
                _ax.set_title(f"{k} with max:{dataframe[k].max():.3f}, min:{dataframe[k].min():.3f}")
            plt.savefig(str(self.save_dir) + f'/{self.save_name}')
        plt.close(fig)


def arg_parser() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description='parser to get parameters to draw csv using matplotlib.')
    parser.add_argument('--csv_path', type=str, required=True, help='csv path to draw.')
    parser.add_argument('--columns_to_draw', required=True, type=str, nargs='+',
                        metavar='columnA, columnB, ...',
                        help='Columns to draw')
    parser.add_argument('--save_dir', required=True, type=str, help='save_dir for the plot.')
    parser.add_argument('--save_name', type=str, default='plot.png', help='Name of the plot.')
    parser.add_argument('--figsize', type=int, nargs=2, default=[10, 15], metavar='A B',
                        help='figure size, default [10, 15]')
    parser.add_argument('--overlap', action='store_true', default=False,
                        help='Whether overlap plots by columns.')
    parser.add_argument('--verbose', action='store_true', help='show args information', default=False)
    args = parser.parse_args()
    if args.verbose:
        print('Arguments:')
        pprint(args)
    return args


if __name__ == '__main__':
    args = arg_parser()
    assert Path(args.csv_path).exists() and Path(args.csv_path).is_file(), args.csv_path

    csv_drawer = DrawCSV(columns_to_draw=args.columns_to_draw,
                         save_dir=args.save_dir,
                         save_name=args.save_name,
                         figsize=args.figsize)

    csv_dataframe = pd.read_csv(args.csv_path)
    csv_drawer.draw(csv_dataframe, together=args.overlap)
