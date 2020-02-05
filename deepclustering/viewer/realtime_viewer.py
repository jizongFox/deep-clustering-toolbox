# this is the viewer script for 3D volumns visualization
from itertools import repeat
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch

Tensor = Union[np.ndarray, torch.Tensor]


class Crop:
    def __init__(self, size=0) -> None:
        super().__init__()
        self.size = size

    def __call__(self, slices: Tensor) -> Tensor:
        H, W, *_ = slices.shape

        return slices[
            self.size // 2 : H - self.size // 2, self.size // 2 : W - self.size // 2
        ]


class Identical:
    def __call__(self, slices):
        return slices


class PLTViewer(object):
    def __init__(self, *keys) -> None:
        try:
            self.remove_keymap_conflicts(set(keys))
        except Exception as e:
            print(e)

    @staticmethod
    def remove_keymap_conflicts(new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith("keymap."):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)


class multi_slice_viewer(PLTViewer):
    def __init__(
        self,
        img_cmap=None,
        mask_cmp=None,
        crop=None,
        is_contour=True,
        alpha=0.5,
        up_key="j",
        down_key="k",
    ) -> None:
        try:
            import matplotlib

            matplotlib.use("qt5agg")
        except Exception as e:
            print(e)

        self.img_cmap = img_cmap
        self.mask_cmap = mask_cmp
        self.crop_func = Crop(crop) if crop is not None else Identical()
        self.up_key = up_key
        self.down_key = down_key
        self.is_contour = is_contour
        self.alpha = alpha
        super().__init__(up_key, down_key)

    def __call__(self, img_volume: Tensor, *gt_volumes: Tensor):
        img_volume = img_volume.squeeze()
        assert len(img_volume.shape) in (
            3,
            4,
        ), f"Only accept 3 or 4 dimensional data, given {len(img_volume.shape)}"

        if len(gt_volumes) > 0:
            B, H, W = img_volume.shape[0:3]
            _B, *_, _H, _W = gt_volumes[0].shape
            assert (
                B == _B and H == _H and W == _W
            ), f"img.shape: {img_volume.shape} and gt_shape: {gt_volumes.shape}"
        fig, axs = plt.subplots(1, max(len(gt_volumes), 1))
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        fig.canvas.mpl_connect("key_press_event", self.process_key)
        fig.canvas.mpl_connect("scroll_event", self.process_mouse_wheel)

        for i, (ax, volume) in enumerate(
            zip(axs, repeat(None) if len(gt_volumes) == 0 else list(gt_volumes))
        ):
            ax.gt_volume = volume
            ax.img_volume = img_volume
            ax.index = img_volume.shape[0] // 2
            ax.imshow(self.crop_func(ax.img_volume[ax.index]), cmap=self.img_cmap)
            if volume is not None:
                ax.con = (
                    ax.contour(
                        self.crop_func(ax.gt_volume[ax.index]), cmap=self.mask_cmap
                    )
                    if self.is_contour
                    else ax.contourf(
                        self.crop_func(ax.gt_volume[ax.index]),
                        cmap=self.mask_cmap,
                        alpha=self.alpha,
                    )
                )
            ax.set_title(f"plane = {ax.index}")
            ax.axis("off")
        plt.show()

    def process_mouse_wheel(self, event):
        fig = event.canvas.figure
        for i, ax in enumerate(fig.axes):
            if event.button == "up":
                self.previous_slice(ax)
            elif event.button == "down":
                self.next_slice(ax)
        fig.canvas.draw()

    def process_key(self, event):
        # using registered key
        fig = event.canvas.figure
        for i, ax in enumerate(fig.axes):
            if event.key == self.up_key:
                self.previous_slice(ax)
            elif event.key == self.down_key:
                self.next_slice(ax)
        fig.canvas.draw()

    def previous_slice(self, ax):
        img_volume = ax.img_volume
        if ax.gt_volume is not None:
            for con in ax.con.collections:
                con.remove()
        ax.index = (ax.index - 1) if (ax.index - 1) >= 0 else 0  # wrap around using %
        ax.images[0].set_array(self.crop_func(img_volume[ax.index]))
        if ax.gt_volume is not None:
            ax.con = (
                ax.contour(self.crop_func(ax.gt_volume[ax.index]), cmap=self.mask_cmap)
                if self.is_contour
                else ax.contourf(
                    self.crop_func(ax.gt_volume[ax.index]),
                    cmap=self.mask_cmap,
                    alpha=self.alpha,
                )
            )
        ax.set_title(f"plane = {ax.index}")

    def next_slice(self, ax):
        img_volume = ax.img_volume
        if ax.gt_volume is not None:
            for con in ax.con.collections:
                con.remove()
        ax.index = (
            (ax.index + 1)
            if (ax.index + 1) < img_volume.shape[0]
            else img_volume.shape[0] - 1
        )
        ax.images[0].set_array(self.crop_func(img_volume[ax.index]))
        if ax.gt_volume is not None:
            ax.con = (
                ax.contour(self.crop_func(ax.gt_volume[ax.index]), cmap=self.mask_cmap)
                if self.is_contour
                else ax.contourf(
                    self.crop_func(ax.gt_volume[ax.index]),
                    cmap=self.mask_cmap,
                    alpha=self.alpha,
                )
            )
        ax.set_title(f"plane = {ax.index}")


def multi_slice_viewer_debug(
    img_volume: Tensor, *gt_volumes: Tensor, no_contour=False, block=False, alpha=0.2
) -> None:
    try:
        import matplotlib

        matplotlib.use("tkagg", force=True)
    except Exception as e:
        print(e)

    def process_mouse_wheel(event):
        fig = event.canvas.figure
        for i, ax in enumerate(fig.axes):
            if event.button == "up":
                previous_slice(ax)
            elif event.button == "down":
                next_slice(ax)
        fig.canvas.draw()

    def process_key(event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == "j":
            previous_slice(ax)
        elif event.key == "k":
            next_slice(ax)
        fig.canvas.draw()

    def previous_slice(ax):
        img_volume = ax.img_volume
        if ax.gt_volume is not None:
            if not no_contour:
                for con in ax.con.collections:
                    con.remove()
            else:
                ax.con.remove()
        ax.index = (ax.index - 1) if (ax.index - 1) >= 0 else 0  # wrap around using %
        ax.images[0].set_array(img_volume[ax.index])
        if ax.gt_volume is not None:
            if volume is not None:
                if not no_contour:
                    ax.con = ax.contour(ax.gt_volume[ax.index])
                else:
                    ax.con = ax.imshow(
                        ax.gt_volume[ax.index], alpha=alpha, cmap="rainbow"
                    )
        ax.set_title(f"plane = {ax.index}")

    def next_slice(ax):
        img_volume = ax.img_volume
        if ax.gt_volume is not None:
            if not no_contour:
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
        if ax.gt_volume is not None:
            if not no_contour:
                ax.con = ax.contour(ax.gt_volume[ax.index])
            else:
                ax.con = ax.imshow(ax.gt_volume[ax.index], alpha=alpha, cmap="rainbow")
        ax.set_title(f"plane = {ax.index}")

    # img_volume = img_volume.squeeze()
    gt_volumes = list(gt_volumes)
    if isinstance(img_volume, torch.Tensor):
        img_volume = img_volume.cpu()
    for num, gt_volumn in enumerate(gt_volumes):
        if isinstance(gt_volumn, torch.Tensor):
            gt_volumes[num] = gt_volumn.cpu()

    assert len(img_volume.shape) in (
        3,
        4,
    ), f"Only accept 3 or 4 dimensional data, given {len(img_volume.shape)}"

    if len(gt_volumes) > 0:
        B, H, W = img_volume.shape[0:3]
        _B, *_, _H, _W = gt_volumes[0].shape
        assert (
            B == _B and H == _H and W == _W
        ), f"img.shape: {img_volume.shape} and gt_shape: {gt_volumes.shape}"
    fig, axs = plt.subplots(1, max(len(gt_volumes), 1))
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])

    for i, (ax, volume) in enumerate(
        zip(axs, repeat(None) if len(gt_volumes) == 0 else list(gt_volumes))
    ):
        ax.gt_volume = volume
        ax.img_volume = img_volume
        ax.index = img_volume.shape[0] // 2
        ax.imshow(ax.img_volume[ax.index], cmap="gray")
        if volume is not None:
            if not no_contour:
                ax.con = ax.contour(ax.gt_volume[ax.index])
            else:
                ax.con = ax.imshow(ax.gt_volume[ax.index], alpha=alpha, cmap="rainbow")
        ax.set_title(f"plane = {ax.index}")
        ax.axis("off")

    fig.canvas.mpl_connect("key_press_event", process_key)
    fig.canvas.mpl_connect("scroll_event", process_mouse_wheel)
    plt.show(block=block)
