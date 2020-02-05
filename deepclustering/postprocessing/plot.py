from functools import partial

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path
from deepclustering.postprocessing.utils import identical, butter_lowpass_filter

c = ["r", "g", "b", "c", "m", "y", "k", "r", "g", "b", "c", "m", "y", "k"]
s = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-"]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot curves given folders, files, and column names"
    )
    parser.add_argument(
        "--folders", type=str, nargs="+", help="input folders", required=True
    )
    parser.add_argument("--file", type=str, required=True, help="csv name")
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=None,
        help="classes to plot, default plot all.",
    )
    parser.add_argument(
        "--yrange", type=float, nargs=2, default=None, help="Y range for plot"
    )
    parser.add_argument(
        "--xrange",
        type=float,
        nargs=2,
        metavar="N",
        default=None,
        help="X range for plot.",
    )
    parser.add_argument("--out_dir", type=str, default=None, help="output_dir")
    parser.add_argument(
        "--smooth_factor", type=float, default=None, help="smooth factor, default None"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    assert isinstance(args.folders, list)
    assert isinstance(args.file, str)
    if args.classes is not None:
        assert isinstance(args.classes, (list))

    file_paths = [Path(p) / args.file for p in args.folders]
    filter = identical
    if args.smooth_factor:
        filter = partial(
            butter_lowpass_filter, cutoff=5000 * args.smooth_factor, fs=10000
        )

    if args.out_dir is not None:
        parent_path = Path(args.out_dir)
    else:
        parent_path = file_paths[0].parents[1]

    for p in file_paths:
        assert p.exists(), p

    # in the case args.classes is None:
    if args.classes is None:
        classes = []
        for file_path in file_paths:
            classes.extend(pd.read_csv(file_path, index_col=0).columns.to_list())
        args.classes = list(set(classes))

    for _class in args.classes:
        for file_path in file_paths:
            try:
                file = filter(pd.read_csv(file_path, index_col=0)[_class])
            except KeyError:
                continue
            plt.plot(file, label=file_path.parents[0])
            # file.plot(label=file_path.parents[0])
        plt.legend()
        plt.title(_class)
        plt.grid()
        if args.xrange is not None:
            plt.xlim(args.xrange)
        if args.yrange:
            plt.ylim(args.yrange)
        plt.savefig(Path(parent_path) / (parent_path.stem + _class + ".png"))
        plt.close("all")

    for i, _class in enumerate(args.classes):
        for j, file_path in enumerate(file_paths):
            try:
                file = pd.read_csv(file_path, index_col=0)[_class]
            except KeyError:
                continue
            file.plot(
                label=file_path.parent.stem + f"/{_class}", color=c[i], linestyle=s[j]
            )
        plt.legend()
        plt.title("total")
    plt.grid()
    if args.xrange is not None:
        plt.xlim(args.xrange)
    if args.yrange:
        plt.ylim(args.yrange)
    plt.savefig(Path(parent_path) / (parent_path.stem + "total.png"))
    plt.close("all")


if __name__ == "__main__":
    main(get_args())
