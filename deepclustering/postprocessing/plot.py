import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path

c = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
s = ['-', '--', '-.', ':', '-']


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot curves given folders, files, and column names')
    parser.add_argument('--folders', type=str, nargs='+', help='input folders', required=True)
    parser.add_argument('--file', type=str, required=True, help='csv name')
    parser.add_argument('--classes', type=str, nargs='+', required=True, help='')
    parser.add_argument('--yrange', type=float, nargs=2, default=[0.4, 0.9], help='Y range for plot')
    parser.add_argument('--xrange', type=float, nargs=2, metavar='N', default=None, help='X range for plot.')
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    assert isinstance(args.folders, list)
    assert isinstance(args.file, str)
    assert isinstance(args.classes, list)

    file_paths = [Path(p) / args.file for p in args.folders]
    parent_path = file_paths[0].parents[1]
    for p in file_paths:
        assert p.exists(), p
    for _class in args.classes:
        for file_path in file_paths:
            file = pd.read_csv(file_path, index_col=0)[_class]
            file.plot(label=file_path.parent.stem)
        plt.legend()
        plt.title(_class)
        plt.grid()
        plt.ylim(args.yrange)
        if args.xrange is not None:
            plt.xlim(args.xrange)
        plt.savefig(Path(parent_path) / (parent_path.stem + _class + '.png'))
        plt.close('all')

    for i, _class in enumerate(args.classes):
        for j, file_path in enumerate(file_paths):
            file = pd.read_csv(file_path, index_col=0)[_class]
            file.plot(label=file_path.parent.stem + f'/{_class}', color=c[i], linestyle=s[j])
        plt.legend()
        plt.title('total')
    plt.grid()
    if args.xrange is not None:
        plt.xlim(args.xrange)
    plt.ylim(args.yrange)
    plt.savefig(Path(parent_path) / (parent_path.stem + 'total.png'))
    plt.close('all')


if __name__ == '__main__':
    main(get_args())
