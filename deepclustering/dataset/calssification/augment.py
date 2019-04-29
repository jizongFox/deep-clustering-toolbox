import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image


class Img2Tensor(object):
    r'''' Grey/ color image to tensor with control of include_rgb, include_grey
    '''

    def __init__(self, include_rgb: bool, include_grey: bool) -> None:
        super().__init__()
        assert (include_rgb or include_grey) == True, f'Options must be \
        True for at least one option, given {include_rgb}, {include_grey}'
        self.include_rgb = include_rgb
        self.include_grey = include_grey

    def __call__(self, rgb_img: Image.Image) -> torch.Tensor:
        '''
        :param rgb_img: input PIL image
        :return: image tensor based on the include_gray and include_rgb
        '''

        assert len(np.array(rgb_img).shape) in (2, 3), f'Check data dimension: {np.array(rgb_img).shape}'
        if len(np.array(rgb_img).shape) == 3:
            assert np.array(rgb_img).shape[2] == 3
        isrgb = True if np.array(rgb_img).shape.__len__() == 3 else False
        grey_img = tf.to_grayscale(rgb_img, num_output_channels=1)
        grey_img_tensor = tf.to_tensor(grey_img)
        assert grey_img_tensor.shape[0] == 1
        if not isrgb:
            assert self.include_grey, f'Input grey image, you must set include_grey to be True'
            return grey_img_tensor
        else:  # you have multiple choice of returning data
            rgb_img_tensor = tf.to_tensor(rgb_img)
            assert rgb_img_tensor.shape[0] == 3
            if self.include_rgb and self.include_grey:
                return torch.cat((grey_img_tensor, rgb_img_tensor), dim=0)
            if self.include_grey and not self.include_rgb:
                return grey_img_tensor
            if not self.include_grey and self.include_rgb:
                return rgb_img_tensor
            else:
                raise AttributeError(f'Something wrong here with img, or options.')


class PILCutout(object):
    r'''
    This function remove a box by randomly choose one part within image
    '''

    def __init__(self, min_box: int = None, max_box: int = None, pad_value: int = 0) -> None:
        '''
        :param min_box: minimal box size
        :param max_box: maxinmal box size
        '''
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
