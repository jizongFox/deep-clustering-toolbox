# this file is written to split capture screen to several subplots
# to be used with the screen capture of Viewer.py
import argparse
from pathlib import Path
from pprint import pprint
from typing import *

import numpy as np
from PIL import Image


def get_image_paths(root: str, extension=".png") -> List[Path]:
    image_paths = sorted(Path(root).glob(f"*{extension}"))
    return image_paths


def split_images(image: Image.Image, title) -> List[Image.Image]:
    print(title)
    np_img = np.array(image)[:, :, :3]
    y_summary = np_img.mean(1).mean(1)
    y_summary_diff = np.where(np.diff(y_summary > 175) == True)[0]
    y_points = y_summary_diff[-2:]
    try:
        assert len(y_points) == 2
    except:
        import ipdb

        ipdb.set_trace()
    x_summary = np_img.mean(0).mean(1)
    x_points = np.where(np.diff(x_summary > 175) == True)[0]
    assert x_points.__len__() == 2 * (
        len(title)
    ), f"x_point:{x_points.__len__()}, title:{len(title)}"
    resulting_imgs = []
    for i in range(len(title)):
        resulting_imgs.append(
            image.crop(
                (x_points[2 * i] + 1, y_points[0] + 1, x_points[2 * i + 1], y_points[1])
            )
        )
    return resulting_imgs


def main(args):
    titles = args.titles
    image_paths = get_image_paths(args.folder_path)
    for img_path in image_paths:
        path: Path = Path(img_path).parent / Path(img_path).stem
        path.mkdir(exist_ok=True, parents=True)
        splited_images = split_images(Image.open(img_path), titles)
        for i in range(len(titles)):
            splited_images[i].save(f"{path}/{titles[i]}.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder_path",
        required=True,
        type=str,
        help="folder having those capture of screen images.",
    )
    parser.add_argument(
        "--titles", required=True, type=str, nargs="+", help="titles to be display"
    )

    args = parser.parse_args()
    pprint(args)
    return args


def call_from_cmd():
    import sys, subprocess

    calling_folder = str(subprocess.check_output("pwd", shell=True))
    sys.path.insert(0, calling_folder)
    main(get_args())


if __name__ == "__main__":
    main(get_args())
