import argparse
import re
from pprint import pprint
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from deepclustering.utils import Vectorize, identical, map_
from pathlib2 import Path
from skimage.transform import resize as resize_func

"""
0: Nearest-neighbor
1: Bi-linear (default)
2: Bi-quadratic
3: Bi-cubic
4: Bi-quartic
5: Bi-quintic
"""
# TODO: add cmap so that `0` with full transparency
# TODO: add alias names for each columns
Tensor = Union[np.ndarray, torch.Tensor]


def cmap(cmap_name="viridis", zero_transparent=False):
    from matplotlib.colors import ListedColormap

    cmap = plt.cm.get_cmap(cmap_name)
    my_cmap = cmap(np.arange(cmap.N))
    if zero_transparent:
        my_cmap[0, -1] = 0
    my_cmap = ListedColormap(my_cmap)
    return my_cmap


def get_parser() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="3D volumne viewer for 2D-sliced images.",
        description="Group and view 2D images with different masks.",
    )
    img_setting_group = parser.add_argument_group(
        "image setting group", "set image folders to the program"
    )
    img_setting_group.add_argument(
        "--img_source",
        type=str,
        required=True,
        help="2D image source folder as the background.",
    )
    img_setting_group.add_argument(
        "--gt_folders", type=str, nargs="*", default=[], help=""
    )
    batch_generation_group = parser.add_argument_group(
        "batch generation group", "group image to batches"
    )
    batch_generation_group.add_argument(
        "--n_subject",
        type=int,
        default=2,
        help="How many subjects you want to display in one figure (default=2).",
    )
    batch_generation_group.add_argument(
        "--shuffle", action="store_true", help="Shuffle the patients."
    )
    batch_generation_group.add_argument(
        "--crop", type=int, default=0, help="Crop image size (default=0)."
    )
    batch_generation_group.add_argument(
        "--group_pattern", type=str, default="patient\d+_\d+", help="group_pattern"
    )
    batch_generation_group.add_argument(
        "--img_extension",
        type=str,
        default="png",
        help="Image extension to select, default='png'",
    )
    batch_generation_group.add_argument(
        "--mapping",
        type=yaml.load,
        nargs="*",
        default=[],
        metavar="{0: 1, 1: 2}",
        help="mapping setting, like:[{0: 1, 1: 2}, {0: 1, 1: 2}] ",
    )
    batch_generation_group.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=None,
        metavar=("M", "N"),
        help="useful option to deal with the cases where imgs, or gts have different resolution.",
    )
    batch_plot_group = parser.add_argument_group(
        "batch plot group", "batch plot parameters"
    )
    batch_plot_group.add_argument(
        "--cmap_name", default="viridis", type=str, help="cmap, default: viridis."
    )
    batch_plot_group.add_argument(
        "--alpha",
        default=1,
        type=float,
        help="transparent factor for ground truth display",
    )
    batch_plot_group.add_argument(
        "--no_contour", default=False, action="store_true", help="show no_contour image"
    )
    batch_plot_group.add_argument(
        "--zeroclass_transparent",
        default=False,
        action="store_true",
        help="set the first class as transparent when no_contour=True",
    )
    args = parser.parse_args()
    args.plot_parameters = {
        "no_contour": args.no_contour,
        "alpha": args.alpha,
        "cmap": {
            "cmap_name": args.cmap_name,
            "zero_transparent": args.zeroclass_transparent,
        },
    }
    return args


class Volume(object):
    """
    This class abstracts the necessary components, such as paired image and masks' paths, grouping paths to subjects.
    :parameter  img_folder
    :type str
    :parameter mask_folder
    :type List[str]
    :interface: __next__() and __cache__()
    :returns img matrix b h w and List[(b h w)]
    """

    def __init__(
        self,
        img_folder: str,
        mask_folder_list: List[str],
        group_pattern=r"patient\d+_\d+",
        img_extension: str = "png",
        crop: int = 0,
        mapping=None,
        resize=None,
    ) -> None:
        super().__init__()
        self.img_folder: Path = Path(img_folder)
        self.crop = crop
        self.resize = resize
        assert self.img_folder.exists() and self.img_folder.is_dir(), self.img_folder
        self.mask_folder_list = [Path(m) for m in mask_folder_list]
        self.num_mask = len(self.mask_folder_list)
        if self.num_mask > 0:
            for m in self.mask_folder_list:
                assert m.exists() and m.is_dir(), m
        self.group_pattern: str = group_pattern
        self.img_extension = img_extension
        self.img_paths, self.mask_paths_dict = self._load_images(
            self.img_folder, self.mask_folder_list, self.img_extension
        )
        self.img_paths_group, self.mask_paths_group_dict = self._group_images(
            self.img_paths, self.mask_paths_dict, self.group_pattern
        )
        assert self.img_paths_group.keys() == self.mask_paths_group_dict.keys()
        for subject, paths in self.img_paths_group.items():
            for m, m_paths in self.mask_paths_group_dict[subject].items():
                assert map_(lambda x: x.stem, paths) == map_(lambda x: x.stem, m_paths)
        print(
            f"Found {len(self.img_paths_group.keys())} subjects with totally {len(self.img_paths)} images."
        )
        self.identifies: List[str] = list(self.img_paths_group.keys())
        print(f"identifies: {self.identifies[:5]}...")
        self.current_identify = 0
        self.img_source: np.ndarray
        self.mask_dicts: Dict[str, np.ndarray]
        if len(mapping) > 0:

            # case 1: there is no gt_folders:
            if len(mask_folder_list) == 0:
                raise RuntimeError(
                    f"No mask is given, you do not need to have mapping, given {mapping}"
                )
            # case 2: all gt_folder has the same mapping
            elif len(mapping) == 1 and len(mask_folder_list) > 1:
                print(f"Found an unique mapping: {mapping}, applying to all gt_masks")
                self.mapping_modules: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
                    f"{str(m)}": np.vectorize(lambda x: mapping[0].get(x))
                    for m in self.mask_folder_list
                }
            elif len(mapping) == len(mask_folder_list) and len(mask_folder_list) >= 2:
                print(
                    f"Found {len(mapping)} mapping: {mapping}, applying to {len(mask_folder_list)} gt_masks"
                )
                self.mapping_modules: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
                    f"{str(m)}": Vectorize(mapping_)
                    for m, mapping_ in zip(self.mask_folder_list, mapping)
                }
            else:
                raise NotImplementedError("mapping logic is wrong.")
        else:
            # no mapping is needed, use identical mapping
            self.mapping_modules: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
                "original image": identical
            }
            self.mapping_modules.update(
                {f"{m}": identical for m in self.mask_folder_list}
            )

        self.img_source, self.mask_dicts = self._preload_subjects(
            self.img_paths_group[self.identifies[self.current_identify]],
            self.mask_paths_group_dict[self.identifies[self.current_identify]],
            self.mapping_modules,
        )

    def __next__(self):
        if self.current_identify < len(self.identifies) - 1:
            self.current_identify += 1
            self.img_source, self.mask_dicts = self._preload_subjects(
                self.img_paths_group[self.identifies[self.current_identify]],
                self.mask_paths_group_dict[self.identifies[self.current_identify]],
                self.mapping_modules,
            )
            if self.crop > 0:
                self.img_source = self.img_source[
                    :, self.crop : -self.crop, self.crop : -self.crop
                ]
                self.mask_dicts = {
                    k: v[:, self.crop : -self.crop, self.crop : -self.crop]
                    for k, v in self.mask_dicts.items()
                }
        print(f"current identify num:{self.current_identify}")
        return self.img_source, self.mask_dicts, self.identifies[self.current_identify]

    def __cache__(self):
        if self.current_identify > 1:
            self.current_identify -= 1
            self.img_source, self.mask_dicts = self._preload_subjects(
                self.img_paths_group[self.identifies[self.current_identify]],
                self.mask_paths_group_dict[self.identifies[self.current_identify]],
                self.mapping_modules,
            )
            if self.crop > 0:
                self.img_source = self.img_source[
                    :, self.crop : -self.crop, self.crop : -self.crop
                ]
                self.mask_dicts = {
                    k: v[:, self.crop : -self.crop, self.crop : -self.crop]
                    for k, v in self.mask_dicts.items()
                }
        print(f"current identify num:{self.current_identify}")

        return self.img_source, self.mask_dicts, self.identifies[self.current_identify]

    @staticmethod
    def _load_images(
        img_folder: Path, mask_folder_list: List[Path], img_extension: str
    ) -> Tuple[List[Path], Dict[str, List[Path]]]:
        img_paths: List[Path] = list(
            img_folder.rglob(f'**/*.{img_extension.replace(".", "")}')
        )
        assert (
            len(img_paths) > 0
        ), f"The length of the image must be higher than 1, given {len(img_paths)}."
        mask_paths_dict: Dict[str, List[Path]] = {}

        for m in mask_folder_list:
            mask_paths_dict[str(m)] = list(
                m.rglob(f'**/*.{img_extension.replace(".", "")}')
            )
            assert (
                mask_paths_dict[str(m)].__len__() > 0
            ), f"The length of the masks {m} must be higher than 1, \
        given {mask_paths_dict[str(m)].__len__()}"
        if mask_folder_list.__len__() > 0:
            assert len(img_paths) == list(mask_paths_dict.values())[0].__len__()
            assert len(set(map(len, list(mask_paths_dict.values())))) == 1, len(
                set(map(len, list(mask_paths_dict.values())))
            )
        return img_paths, mask_paths_dict

    @staticmethod
    def _group_images(
        img_paths: List[Path],
        mask_paths_dict: Dict[str, List[Path]],
        group_pattern: str,
    ) -> Tuple[Dict[str, List[Path]], Dict[str, Dict[str, List[Path]]]]:
        img_stem: List[str] = [i.stem for i in img_paths]
        r_pattern = re.compile(group_pattern)
        img_identification = sorted(
            [r_pattern.match(i_stem).group(0) for i_stem in img_stem]
        )  # type: ignore
        unique_identification = sorted(list(set(img_identification)))
        # grouping
        img_paths_group: Dict[str, List[Path]] = {}
        mask_paths_group: Dict[str, Dict[str, List[Path]]] = {}
        for identy in unique_identification:
            img_paths_group[identy] = sorted(
                [i for i in img_paths if re.compile(identy).search(str(i)) is not None]
            )
            mask_paths_group[identy] = {}
            for m, m_list in mask_paths_dict.items():
                mask_paths_group[identy][m] = sorted(
                    [i for i in m_list if re.compile(identy).search(str(i)) is not None]
                )
        return img_paths_group, mask_paths_group

    def _preload_subjects(
        self,
        batch_image_path: List[Path],
        batch_mask_paths_dict: Dict[str, List[Path]],
        mapping_module,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        img_source: np.ndarray = np.stack(
            [np.array(Image.open(str(x)).convert("P")) for x in batch_image_path],
            axis=0,
        )
        mask_source_dict: Dict[str, np.ndarray] = {}
        if len(batch_mask_paths_dict) > 0:
            for k, v in batch_mask_paths_dict.items():
                # we have images as ground truth
                mask_source_dict[k] = np.stack(
                    [np.array(Image.open(str(x)).convert("P")) for x in v], axis=0
                )
                mask_source_dict[k] = mapping_module[k](mask_source_dict[k])
                # assert mask_source_dict[k].shape == img_source.shape
        else:  # we have no ground truth.
            mask_source_dict["original image"] = np.zeros_like(img_source)
        if self.resize:
            img_source, mask_source_dict = self.optional_resize(
                img_source, mask_source_dict, self.resize
            )

        return img_source, mask_source_dict

    @staticmethod
    def optional_resize(
        img_source_np_tensor: np.ndarray,
        mask_source_np_dict: Dict[str, np.ndarray],
        resize=[256, 256],
    ):
        if isinstance(resize, int):
            resize = (resize, resize)
        elif isinstance(resize, list):
            for _resize in resize:
                isinstance(_resize, int)
        assert isinstance(resize, list) and len(resize) == 2

        def _if_need_resized(matrix: np.ndarray, resize: List[int] = resize) -> bool:
            assert matrix.shape.__len__() == 3
            return matrix.shape[1:] != tuple(resize)

        # bi-linear resize
        resized_img_tensor = (
            img_source_np_tensor
            if not _if_need_resized(img_source_np_tensor)
            else np.stack(
                [
                    resize_func(
                        img_slice, output_shape=resize, order=1, preserve_range=True
                    )
                    for img_slice in img_source_np_tensor
                ],
                axis=0,
            )
        )
        # Nearest-neighbor
        resized_mask_source_np_dict = {}
        for k, v in mask_source_np_dict.items():
            resized_mask_source_np_dict[k] = (
                v
                if not _if_need_resized(v)
                else np.stack(
                    [
                        resize_func(
                            mask_slice,
                            output_shape=resize,
                            order=0,
                            preserve_range=True,
                            anti_aliasing=False,
                        )
                        for mask_slice in v
                    ],
                    axis=0,
                )
            )
        return resized_img_tensor, resized_mask_source_np_dict


class Multi_Slice_Viewer(object):
    def __init__(
        self, volume: Volume, n_subject=1, shuffle_subject=False, contour=True, **kwargs
    ) -> None:
        super().__init__()
        self.volume = volume
        self.n_subject = n_subject
        self.show_contour = contour
        self.volume.identifies = (
            np.random.permutation(self.volume.identifies)
            if shuffle_subject
            else self.volume.identifies
        )
        self.kwargs = kwargs

    @staticmethod
    def _preproccess_data(volume_output):
        img_volume, gt_volume_dict, subject_name = volume_output
        mask_volumes = list(gt_volume_dict.values())
        _, _, _ = img_volume.shape
        if not isinstance(mask_volumes, list):
            mask_volumes = [mask_volumes]
        if mask_volumes[0] is not None:
            assert (
                img_volume.shape == mask_volumes[0].shape
            ), f"img_volume.shape:{img_volume.shape}, mask_volumes[0].shape: {mask_volumes[0].shape}"
        mask_names = list(gt_volume_dict.keys())
        return img_volume, mask_volumes, subject_name, mask_names

    def show(self,):
        fig, axs = plt.subplots(
            self.n_subject, self.volume.num_mask if self.volume.num_mask > 1 else 1
        )
        axs = np.array([axs]) if not isinstance(axs, np.ndarray) else axs
        # fixme: error raised if there is no ground truth
        self.axs = axs.reshape(
            (self.n_subject, self.volume.num_mask if self.volume.num_mask > 1 else 1)
        )

        for row in range(self.n_subject):
            img_volume, mask_volumes, subject_name, mask_names = self._preproccess_data(
                self.volume.__next__()
            )
            for i, (ax, mask_volume, mask_name) in enumerate(
                zip(self.axs[row], mask_volumes, mask_names)
            ):
                ax.subject_name = subject_name
                ax.mask_name = mask_name
                ax.mask_volume = mask_volume
                ax.img_volume = img_volume
                ax.index = img_volume.shape[0] // 2
                ax.imshow(ax.img_volume[ax.index], cmap="gray")
                if self.show_contour:
                    ax.con = ax.contour(
                        ax.mask_volume[ax.index],
                        alpha=self.kwargs["plot_parameters"]["alpha"],
                    )
                else:
                    ax.con = ax.imshow(
                        ax.mask_volume[ax.index],
                        cmap=cmap(**self.kwargs["plot_parameters"]["cmap"]),
                        alpha=self.kwargs["plot_parameters"]["alpha"],
                    )
                ax.axis("off")
                ax.set_title(f"{subject_name} @ plane:{ax.index} with {mask_name}")

        fig.canvas.mpl_connect("scroll_event", self.process_mouse_wheel)
        fig.canvas.mpl_connect("button_press_event", self.process_mouse_button)
        # plt.tight_layout()
        # mng = plt.get_current_fig_manager()
        # mng.full_screen_toggle()
        plt.show()

    def process_mouse_wheel(self, event):
        fig = event.canvas.figure
        for i, ax in enumerate(fig.axes):
            if event.button == "up":
                self._previous_slice(
                    ax,
                    self.show_contour,
                    self.kwargs["plot_parameters"]["cmap"],
                    alpha=self.kwargs["plot_parameters"]["alpha"],
                )
            elif event.button == "down":
                self._next_slice(
                    ax,
                    self.show_contour,
                    self.kwargs["plot_parameters"]["cmap"],
                    alpha=self.kwargs["plot_parameters"]["alpha"],
                )
        fig.canvas.draw()

    def process_mouse_button(self, event):
        fig = event.canvas.figure
        if event.button in (1, 3):
            if event.button == 1:
                self._change_subject(mode=self.volume.__next__)
            elif event.button == 3:
                self._change_subject(mode=self.volume.__cache__)
            fig.canvas.draw()

    def _change_subject(self, mode):
        axs = self.axs
        for row in range(self.n_subject):
            img_volume, mask_volumes, subject_name, mask_names = self._preproccess_data(
                mode()
            )
            for i, (ax, mask_volume, mask_name) in enumerate(
                zip(axs[row], mask_volumes, mask_names)
            ):
                ax.clear()
                ax.subject_name = subject_name
                ax.mask_name = mask_name
                ax.mask_volume = mask_volume
                ax.img_volume = img_volume
                ax.index = img_volume.shape[0] // 2
                ax.imshow(ax.img_volume[ax.index], cmap="gray")
                if self.show_contour:
                    ax.con = ax.contour(
                        ax.mask_volume[ax.index],
                        alpha=self.kwargs["plot_parameters"]["alpha"],
                    )
                else:
                    ax.con = ax.imshow(
                        ax.mask_volume[ax.index],
                        cmap=cmap(**self.kwargs["plot_parameters"]["cmap"]),
                        alpha=self.kwargs["plot_parameters"]["alpha"],
                    )
                ax.axis("off")
                ax.set_title(f"{subject_name} @ plane:{ax.index} with {mask_name}")

    @staticmethod
    def _previous_slice(ax, show_contour, cmap_params={}, alpha=0.5):
        img_volume = ax.img_volume
        if show_contour:
            for con in ax.con.collections:
                con.remove()
        else:
            ax.con.remove()
        ax.index = (ax.index - 1) if (ax.index - 1) >= 0 else 0  # wrap around using %
        ax.images[0].set_array(img_volume[ax.index])
        if show_contour:
            ax.con = ax.contour(ax.mask_volume[ax.index])
        else:
            ax.con = ax.imshow(
                ax.mask_volume[ax.index], cmap=cmap(**cmap_params), alpha=alpha
            )
        ax.set_title(f"{ax.subject_name} @ plane:{ax.index} with {ax.mask_name}")

    @staticmethod
    def _next_slice(ax, show_contour, cmap_params={}, alpha=0.5):
        img_volume = ax.img_volume
        if show_contour:
            for con in ax.con.collections:
                con.remove()
        else:
            ax.con.remove()
        ax.index = (
            (ax.index + 1)
            if (ax.index + 1) < img_volume.shape[0]
            else img_volume.shape[0] - 1
        )
        ax.images[0].set_array(img_volume[ax.index])
        if show_contour:
            ax.con = ax.contour(ax.mask_volume[ax.index])
        else:
            ax.con = ax.imshow(
                ax.mask_volume[ax.index], cmap=cmap(**cmap_params), alpha=alpha
            )
        ax.set_title(f"{ax.subject_name} @ plane:{ax.index} with {ax.mask_name}")


def main():
    import subprocess, sys

    current_folder = subprocess.check_output("pwd", shell=True).strip().decode()
    sys.path.insert(0, current_folder)
    print(current_folder)
    args = get_parser()
    pprint(vars(args))
    V = Volume(
        args.img_source,
        args.gt_folders,
        group_pattern=args.group_pattern,
        img_extension=args.img_extension,
        crop=args.crop,
        mapping=args.mapping,
        resize=args.resize,
    )

    Viewer = Multi_Slice_Viewer(
        V,
        shuffle_subject=args.shuffle,
        n_subject=args.n_subject,
        contour=not args.no_contour,
        plot_parameters=args.plot_parameters,
    )
    Viewer.show()


if __name__ == "__main__":
    """
    python Viewer.py --img_source=../../.data/ACDC-all/val/img \
    --group_pattern=patient\d+_\d+\_\d+ \
    --gt_folders ../../.data/ACDC-all/val/gt \
    ../../.data/ACDC-all/val/gt1 \
    ../../.data/ACDC-all/val/gt2 \
    --resize 512 512 \
    --mapping "{0: 0, 1: 1, 2: 2, 3: 3}" \
    --crop=40
    """

    args = get_parser()
    pprint(args)
    V = Volume(
        args.img_source,
        args.gt_folders,
        group_pattern=args.group_pattern,
        img_extension=args.img_extension,
        crop=args.crop,
        mapping=args.mapping,
        resize=args.resize,
    )

    Viewer = Multi_Slice_Viewer(
        V,
        shuffle_subject=args.shuffle,
        n_subject=args.n_subject,
        contour=not args.no_contour,
        plot_parameters=args.plot_parameters,
    )
    Viewer.show()
