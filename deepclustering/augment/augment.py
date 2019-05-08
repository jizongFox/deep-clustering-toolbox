import collections
import numbers
import random
from typing import *

import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from torch import nn

__all__ = ["Img2Tensor", "PILCutout", "RandomCrop", "Resize", "CenterCrop"]

Iterable = collections.abc.Iterable

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


class Img2Tensor(object):
    r""" Grey/ color image to tensor with control of include_rgb, include_grey
    """

    def __init__(self, include_rgb: bool = False, include_grey: bool = True) -> None:
        super().__init__()
        assert (include_rgb or include_grey), f'Options must be \
        True for at least one option, given {include_rgb}, {include_grey}'
        self.include_rgb = include_rgb
        self.include_grey = include_grey

    def __call__(self, rgb_img: Image.Image) -> torch.Tensor:
        r"""
        :param rgb_img: input PIL image
        :return: image tensor based on the include_gray and include_rgb
        """

        assert len(np.array(rgb_img).shape) in (2, 3), f'Check data dimension:' \
            f' {np.array(rgb_img).shape}'  # type: ignore
        if len(np.array(rgb_img).shape) == 3:
            assert np.array(rgb_img).shape[2] == 3
        isrgb: bool = np.array(rgb_img).shape.__len__() == 3
        grey_img = tf.to_grayscale(rgb_img, num_output_channels=1)
        grey_img_tensor = tf.to_tensor(grey_img)
        assert grey_img_tensor.shape[0] == 1
        if not isrgb:
            assert self.include_grey, f'Input grey image, you must set include_grey to be True'
            return grey_img_tensor
        # else:  # you have multiple choice of returning data
        rgb_img_tensor = tf.to_tensor(rgb_img)
        assert rgb_img_tensor.shape[0] == 3
        if self.include_rgb and self.include_grey:
            return torch.cat((grey_img_tensor, rgb_img_tensor), dim=0)  # pylint: ignore
        if self.include_grey and not self.include_rgb:
            return grey_img_tensor
        if not self.include_grey and self.include_rgb:
            return rgb_img_tensor
        raise AttributeError(f'Something wrong here with img, or options.')


class PILCutout(object):
    r"""
    This function remove a box by randomly choose one part within image
    """

    def __init__(self, min_box: int, max_box: int, pad_value: int = 0) -> None:
        r"""
        :param min_box: minimal box size
        :param max_box: maxinmal box size
        """
        super().__init__()
        self.min_box = int(min_box)
        self.max_box = int(max_box)
        self.pad_value = int(pad_value)

    def __call__(self, img: Image.Image) -> Image.Image:
        r_img = img.copy()
        w, h = img.size
        # find left, upper, right, lower
        box_sz = np.random.randint(self.min_box, self.max_box + 1)
        half_box_sz = int(np.floor(box_sz / 2.))
        x_c = np.random.randint(half_box_sz, w - half_box_sz)
        y_c = np.random.randint(half_box_sz, h - half_box_sz)
        box = (
            x_c - half_box_sz,
            y_c - half_box_sz,
            x_c + half_box_sz,
            y_c + half_box_sz
        )
        r_img.paste(self.pad_value, box=box)
        return r_img


class RandomCrop(object):
    """Crop the given PIL Image at a random location.

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

    def __init__(self, size: Union[int, Tuple[int, int], List[int]],
                 padding: Union[int, Tuple[int, int, int, int], List[int]] = None,
                 pad_if_needed: bool = False,
                 fill: Union[int, float] = 0,
                 padding_mode: str = 'constant'):
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
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding is not None:
            img = tf.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = tf.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = tf.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return tf.crop(img, i, j, h, w)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


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

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        return tf.resize(img, self.size, self.interpolation)

    def __repr__(self) -> str:
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, ' \
                                         'interpolation={1})'.format(self.size, interpolate_str)


class CenterCrop(object):
    """Crops the given Tensor Image at the center.

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

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return tf.center_crop(img, self.size)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class SobelProcess(object):
    r"""
    This class involves the Sobel Processing to extrait the contour.
    Input image should be torch.Tensor
    """

    def __init__(self, include_origin: bool = False) -> None:
        r"""
        :param include_origin: stack origin tensor in the end
        """
        super().__init__()

        self.include_origin = include_origin
        sobel1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).astype(np.float)
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1.weight = nn.Parameter(torch.Tensor(sobel1).unsqueeze(0).unsqueeze(0))
        sobel2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2.weight = nn.Parameter(torch.Tensor(sobel2).unsqueeze(0).unsqueeze(0))

        csobel1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).astype(np.float)
        self.cconv1 = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.cconv1.weight = nn.Parameter(torch.Tensor(csobel1).unsqueeze(0).expand((1, 3, 3, 3)) / 3.0)
        csobel2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float)
        self.cconv2 = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.cconv2.weight = nn.Parameter(torch.Tensor(csobel2).unsqueeze(0).expand((1, 3, 3, 3)) / 3.0)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        assert isinstance(img, torch.Tensor)
        b, c, h, w = img.shape
        assert c in (1, 3), f'Image channel should be 1 or 3, given {c}.'
        if c == 3:
            import warnings
            warnings.warn('only C = 1 supported for sobel filtering, given {}.'.format(c))

        dx: torch.Tensor = self.conv1(img).detach() if c == 1 \
            else self.cconv1(img).detach()
        dy: torch.Tensor = self.conv2(img).detach() if c == 1 \
            else self.cconv2(img).detach()
        sobel_imgs: torch.Tensor = torch.cat((dx, dy), dim=1)
        if not self.include_origin:
            return sobel_imgs
        return torch.cat((img, sobel_imgs), dim=1)

    def to(self, device):
        self.conv1.to(device)
        self.conv2.to(device)
        self.cconv1.to(device)
        self.cconv2.to(device)


class RandomTransforms(object):
    """Base class for a list of transformations with randomness
    Args:
        transforms (list or tuple): list of transformations
    """

    def __init__(self, transforms: Union[Tuple[Callable], List[Callable]]):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    def __call__(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomApply(RandomTransforms):
    """Apply randomly a list of transformations with a given probability

    Args:
        transforms (list or tuple): list of transformations
        p (float): probability
    """

    def __init__(
            self,
            transforms: Union[Tuple[Callable], List[Callable]],
            p: float = 0.5
    ):
        super(RandomApply, self).__init__(transforms)
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:  # type:ignore
        if self.p < random.random():
            return img
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomChoice(RandomTransforms):
    """Apply single transformation randomly picked from a list
    """

    def __call__(self, img):
        t = random.choice(self.transforms)
        return t(img)


class ToLabel():
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, img):
        np_img = np.array(img)[None, ...].astype(np.float32)
        t_img = torch.from_numpy(np_img)
        return t_img.long()