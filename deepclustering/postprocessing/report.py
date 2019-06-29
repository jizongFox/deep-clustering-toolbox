import argparse
from pathlib import Path
from pprint import pprint
from typing import List

import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description="getting final results")
    parser.add_argument(
        "--folder", type=str, required=True, help="folder path, only one folder"
    )
    parser.add_argument("--file", type=str, required=True, help="csv to report.")
    parser.add_argument(
        "--classes", type=str, nargs="+", required=True, help="classes to report"
    )

    return parser.parse_args()


def main(args):
    file_paths: List[Path] = list(Path(args.folder).glob(f"**/{args.file}"))
    print("Found file lists:")
    pprint(file_paths)
    results = {}
    for f in file_paths:
        record = pd.read_csv(f, index_col=0)[args.classes].max().to_dict()
        results[str(f.parent)] = record
    print(pd.DataFrame(results).T)
    pd.DataFrame(results).T.to_csv(Path(args.folder) / "summary.csv")


if __name__ == "__main__":
    main(get_args())
