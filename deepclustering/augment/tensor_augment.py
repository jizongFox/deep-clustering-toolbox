"""
Data augmentation given only the high dimensional Tensors, instead of PIL images.
"""
import numbers
import random
from typing import *

import numpy as np
import torch
from torch.nn import functional as F

T = Union[np.ndarray, torch.Tensor]
_Tensor = (np.ndarray, torch.Tensor)


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class TensorCutout(object):
    r"""
    This function remove a box by randomly choose one part within image Tensor
    """

    def __init__(
        self, min_box: int, max_box: int, pad_value: Union[int, float] = 0
    ) -> None:
        r"""
        :param min_box: minimal box size
        :param max_box: maxinmal box size
        """
        super().__init__()
        self.min_box = int(min_box)
        self.max_box = int(max_box)
        self.pad_value = pad_value

    def __call__(self, img_tensor: T) -> T:
        assert isinstance(img_tensor, _Tensor)
        b, c, h, w = img_tensor.shape
        r_img_tensor = (
            img_tensor.copy()
            if isinstance(img_tensor, np.ndarray)
            else img_tensor.clone()
        )
        # find left, upper, right, lower
        box_sz = np.random.randint(self.min_box, self.max_box + 1)
        half_box_sz = int(np.floor(box_sz / 2.0))
        x_c = np.random.randint(half_box_sz, w - half_box_sz)
        y_c = np.random.randint(half_box_sz, h - half_box_sz)
        box = (
            x_c - half_box_sz,
            y_c - half_box_sz,
            x_c + half_box_sz,
            y_c + half_box_sz,
        )
        r_img_tensor[:, :, box[1] : box[3], box[0] : box[2]] = 0
        return r_img_tensor


class RandomCrop(object):
    """Crop the given Tensor Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge,
        reflect or symmetric. Default is constant.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int], List[int]],
        padding: Union[
            int, Tuple[int, int], Tuple[int, int, int, int], List[int]
        ] = None,
        pad_if_needed: bool = False,
        fill: Union[int, float] = 0,
        padding_mode: str = "constant",
    ):
        if isinstance(size, numbers.Number):
            self.size: Tuple[int, int] = (int(size), int(size))
        else:
            self.size: Tuple[int, int] = size  # type: ignore
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        b, c, w, h = img.shape
        tw, th = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img: T) -> T:
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        assert isinstance(img, _Tensor)
        b, c, h, w = img.shape
        r_img = img.copy() if isinstance(img, np.ndarray) else img.clone()
        if self.padding is not None:
            if isinstance(self.padding, int):
                padding = (self.padding, self.padding)
            elif isinstance(self.padding, tuple) and self.padding.__len__() == 2:
                # padding = self.padding + self.padding
                padding = self.padding

            if isinstance(r_img, np.ndarray):
                r_img = np.pad(
                    r_img,
                    pad_width=((0, 0), (0, 0), padding, padding),
                    constant_values=self.fill,
                    mode=self.padding_mode,
                )
            else:
                r_img = F.pad(r_img, padding)

        # pad the width if needed
        if self.pad_if_needed and r_img.shape[2] < self.size[0]:
            r_img = np.pad(
                r_img,
                pad_width=(
                    (0, 0),
                    (0, 0),
                    (
                        int((self.size[0] - r_img.shape[2]) / 2) + 1,
                        int((self.size[0] - r_img.shape[2]) / 2) + 1,
                    ),
                    (0, 0),
                ),
                constant_values=self.fill,
                mode=self.padding_mode,
            )
            # pad the height if needed
        if self.pad_if_needed and r_img.shape[3] < self.size[1]:
            r_img = np.pad(
                r_img,
                pad_width=(
                    (0, 0),
                    (0, 0),
                    (0, 0),
                    (
                        int((self.size[1] - r_img.shape[3]) / 2) + 1,
                        int((self.size[1] - r_img.shape[3]) / 2) + 1,
                    ),
                ),
                constant_values=self.fill,
                mode=self.padding_mode,
            )

        # todo: set padding as default when the size is larger than the current size.
        i, j, h, w = self.get_params(r_img, self.size)

        return r_img[:, :, int(j) : int(j + w), int(i) : int(i + h)]

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(size={0}, padding={1})".format(
            self.size, self.padding
        )


class Resize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img: T) -> T:
        """
        Args:
            img Tensor Image: Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        # todo speed up the resize.
        assert isinstance(img, _Tensor)
        b, c, h, w = img.shape
        torch_img = (
            torch.Tensor(img.copy()).float()
            if isinstance(img, np.ndarray)
            else img.clone()
        )
        torch_img = F.upsample(
            torch_img,
            size=(self.size[0], self.size[1]),
            mode=self.interpolation,
            align_corners=True,
        )
        if isinstance(img, np.ndarray):
            return torch_img.detach().numpy()
        return torch_img


class CenterCrop(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, numbers.Number):
            self.size: Tuple[int, int] = (int(size), int(size))
        else:
            self.size: Tuple[int, int] = size  # type:ignore

    @staticmethod
    def get_parameter(img, output_size):
        if isinstance(output_size, numbers.Number):
            output_size = (int(output_size), int(output_size))
        _, _, h, w = img.shape
        th, tw = output_size
        i = int(round((h - th) / 2.0))
        j = int(round((w - tw) / 2.0))
        return i, j, th, tw

    def __call__(self, img: T) -> T:
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        assert isinstance(img, _Tensor)
        b, c, h, w = img.shape
        assert (
            h >= self.size[0] and w >= self.size[1]
        ), f"Image size {h} and {w}, given {self.size}."
        r_img = img.copy() if isinstance(img, np.ndarray) else img.clone()
        i, j, th, tw = self.get_parameter(r_img, self.size)

        return r_img[:, :, i : i + th, j : j + tw]

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(size={0})".format(self.size)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5, dim=3):
        self.p = p
        self.dim = dim

    def __call__(self, img):
        """
        Args:
            img (Tensor Image): Image Tensor to be flipped. Must have 4 dimensions

        Returns:
            Tensor Image: Randomly flipped image.
        """
        if random.random() < self.p:
            img = img.flip(self.dim)

        return img

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5, dim=2):
        self.p = p
        self.dim = dim

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            img = img.flip(self.dim)
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class GaussianNoise(object):
    def __init__(self, std=0.15) -> None:
        super().__init__()
        self._std = std

    def __call__(self, img: T) -> T:
        if isinstance(img, torch.Tensor):
            noise = torch.randn_like(img) * self._std
        else:
            noise = np.random.randn(*img.shape) * self._std

        return img + noise
