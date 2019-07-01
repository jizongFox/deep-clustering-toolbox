import math
import random
import warnings
from typing import *

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from torch.utils.data import Dataset
from torchvision import transforms

from deepclustering.augment import TransformInterface
from deepclustering.augment.sychronized_augment import SequentialWrapper
from deepclustering.utils import fix_all_seed
from deepclustering.utils.segmentation import utils

__all__ = [
    "default_toy_img_transform",
    "ShapesDataset",
    "Cls_ShapesDataset",
    "Seg_ShapesDataset",
]


class ShapesDataset(Dataset):
    def __init__(
        self,
        count: int = 1000,
        max_object_per_img: int = 4,
        max_object_scale: float = 0.25,
        height: int = 256,
        width: int = 256,
        transform: Callable = None,
        target_transform: Callable = None,
        seed: int = 0,
    ) -> None:
        """
        Interface for ShapesDataset
        :param count: how many samples to generate
        :param max_object_per_img: how many objects to show in an image
        :param height: image height
        :param width: image height
        """
        super().__init__()
        fix_all_seed(seed)
        assert (
            max_object_per_img >= 1
        ), f"max_object_per_img should be larger than 1, given {max_object_per_img}."
        self.max_object_per_img = max_object_per_img
        assert 0 < max_object_scale <= 0.75
        self.max_object_scale = max_object_scale
        self.height = height
        self.width = width
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}
        # self.transform: Callable = transform
        # self.target_transform: Callable = target_transform
        self.squential_transform = SequentialWrapper(
            img_transform=transform,
            target_transform=target_transform,
            if_is_target=[False, True, True],
        )

        self.add_class("shapes", 1, "square")
        self.add_class("shapes", 2, "circle")
        self.add_class("shapes", 3, "triangle")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(count):
            bg_color, shapes = self.random_image(height, width)
            self.add_image(
                "shapes",
                image_id=i,
                path=None,
                width=width,
                height=height,
                bg_color=bg_color,
                shapes=shapes,
            )

        self.prepare()

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {"id": image_id, "source": source, "path": path}
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info["source"] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({"source": source, "id": class_id, "name": class_name})

    def random_shape(self, height, width):
        """Generates specifications of a random shape that lies within
        the given height and width boundaries.
        Returns a tuple of three valus:
        * The shape name (square, circle, ...)
        * Shape color: a tuple of 3 values, RGB.
        * Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        """
        # Shape
        shape = random.choice(["square", "circle", "triangle"])
        # Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
        # Center x, y
        buffer = min(int(self.height / 20) + 1, int(self.width / 20) + 1)
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(
            buffer,
            random.randint(
                buffer, max(int(height ** self.max_object_scale), buffer + 1)
            ),
        )
        return shape, color, (x, y, s)

    def random_image(self, height, width):
        """Creates random specifications of an image with multiple shapes.
        Returns the background color of the image and a list of shape
        specifications that can be used to draw the image.
        """
        # Pick random background color
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        # Generate a few random shapes and record their
        # bounding boxes
        shapes = []
        boxes = []
        N = random.randint(1, self.max_object_per_img)
        for _ in range(N):
            shape, color, dims = self.random_shape(height, width)
            shapes.append((shape, color, dims))
            x, y, s = dims
            boxes.append([y - s, x - s, y + s, x + s])
        # Apply non-max suppression wit 0.3 threshold to avoid
        # shapes covering each other
        keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.3)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes

    def draw_shape(self, image, shape, dims, color):
        """Draws a shape from the given specs."""
        # Get the center x, y and the size s
        x, y, s = dims
        if shape == "square":
            image = cv2.rectangle(image, (x - s, y - s), (x + s, y + s), color, -1)
        elif shape == "circle":
            image = cv2.circle(image, (x, y), s, color, -1)
        elif shape == "triangle":
            points = np.array(
                [
                    [
                        (x, y - s),
                        (x - s / math.sin(math.radians(60)), y + s),
                        (x + s / math.sin(math.radians(60)), y + s),
                    ]
                ],
                dtype=np.int32,
            )
            image = cv2.fillPoly(image, points, color)
        return image

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        bg_color = np.array(info["bg_color"]).reshape([1, 1, 3])
        image = np.ones([info["height"], info["width"], 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        for shape, color, dims in info["shapes"]:
            image = self.draw_shape(image, shape, dims, color)
        return image

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        shapes = info["shapes"]
        count = len(shapes)
        mask = np.zeros([info["height"], info["width"], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info["shapes"]):
            mask[:, :, i : i + 1] = self.draw_shape(
                mask[:, :, i : i + 1].copy(), shape, dims, 1
            )
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask, class_ids.astype(np.int32)

    def __getitem__(self, index):
        img = self.load_image(index) / 256
        instance_mask, class_ids = self.load_mask(index)
        for i, c in enumerate(class_ids):
            instance_mask[:, :, i][instance_mask[:, :, i] == 1] = c
        global_mask = instance_mask.sum(2)
        for i, c in enumerate(class_ids):
            instance_mask[:, :, i][instance_mask[:, :, i] == c] = i + 1
        instance_mask = instance_mask.sum(2)
        img = transforms.ToTensor()(img).to(torch.float)
        global_mask = utils.ToLabel()(global_mask).to(torch.uint8)
        instance_mask = utils.ToLabel()(instance_mask).to(torch.uint8)

        img, global_mask, instance_mask = self.squential_transform(
            img, global_mask, instance_mask
        )

        return img.float(), global_mask.long(), instance_mask.long()

    def __len__(self):
        return self._image_ids.__len__()

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {
            "{}.{}".format(info["source"], info["id"]): id
            for info, id in zip(self.class_info, self.class_ids)
        }
        self.image_from_source_map = {
            "{}.{}".format(info["source"], info["id"]): id
            for info, id in zip(self.image_info, self.image_ids)
        }

        # Map sources to class_ids they support
        self.sources = list(set([i["source"] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info["source"]:
                    self.source_class_ids[source].append(i)

    @property
    def image_ids(self):
        return self._image_ids


class Cls_ShapesDataset(ShapesDataset):
    def __init__(
        self,
        count: int = 1000,
        max_object_scale: float = 0.25,
        height: int = 256,
        width: int = 256,
        transform=None,
        target_transform=None,
    ) -> None:
        super().__init__(
            count, 1, max_object_scale, height, width, transform, target_transform
        )

    def __getitem__(self, index):
        img, global_mask, instance_mask = super().__getitem__(index)
        # if len(global_mask.unique()) == 2:
        #     warnings.warn(f'Only background and one type of foreground should be presented, \
        #     given {len(global_mask.unique())} type.')
        return (
            img,
            sorted(global_mask.unique())[1].long() - 1,
        )  # remove the background class


class Seg_ShapesDataset(ShapesDataset):
    def __init__(
        self,
        count: int = 1000,
        max_object_per_img=3,
        max_object_scale: float = 0.25,
        height: int = 256,
        width: int = 256,
        transform=None,
        target_transform=None,
    ) -> None:
        super().__init__(
            count,
            max_object_per_img,
            max_object_scale,
            height,
            width,
            transform,
            target_transform,
        )

    def __getitem__(self, index):
        img, global_mask, instance_mask = super().__getitem__(index)
        return img, global_mask.long()


class Ins_ShapesDataset(ShapesDataset):
    def __init__(
        self,
        count: int = 1000,
        max_object_per_img=4,
        max_object_scale: float = 0.25,
        height: int = 256,
        width: int = 256,
        transform=None,
        target_transform=None,
    ) -> None:
        super().__init__(
            count,
            max_object_per_img,
            max_object_scale,
            height,
            width,
            transform,
            target_transform,
        )

    def __getitem__(self, index):
        img, global_mask, instance_mask = super().__getitem__(index)
        return img, global_mask.long(), instance_mask.long()


transform_dict = {
    "tf1": {
        "img": {
            "ToPILImage": {},
            # 'RandomRotation': {'degrees': 25},
            "randomcrop": {"size": (96, 96)},
            "Resize": {"size": (96, 96), "interpolation": 0},
            "Img2Tensor": {"include_rgb": False, "include_grey": True},
        },
        "target": {
            "ToPILImage": {},
            # 'RandomRotation': {'degrees': 25},
            "randomcrop": {"size": (96, 96)},
            "Resize": {"size": (96, 96), "interpolation": 0},
            "ToLabel": {},
        },
    },
    "tf2": {
        "img": {
            "ToPILImage": {},
            # 'RandomRotation': {'degrees': 25},
            "randomcrop": {"size": (96, 96)},
            "Resize": {"size": (96, 96), "interpolation": 0},
            "RandomHorizontalFlip": {"p": 0.5},
            "ColorJitter": {
                "brightness": [0.6, 1.4],
                "contrast": [0.6, 1.4],
                "saturation": [0.6, 1.4],
                "hue": [-0.125, 0.125],
            },
            "Img2Tensor": {"include_rgb": False, "include_grey": True},
        },
        "target": {
            "ToPILImage": {},
            # 'RandomRotation': {'degrees': 25},
            "randomcrop": {"size": (96, 96)},
            "Resize": {"size": (96, 96), "interpolation": 0},
            "RandomHorizontalFlip": {"p": 0.5},
            "ToLabel": {},
        },
    },
    "tf3": {
        "img": {
            "ToPILImage": {},
            "CenterCrop": {"size": (96, 96)},
            "Resize": {"size": (96, 96), "interpolation": 0},
            "Img2Tensor": {"include_rgb": False, "include_grey": True},
        },
        "target": {
            "ToPILImage": {},
            "CenterCrop": {"size": (96, 96)},
            "Resize": {"size": (96, 96), "interpolation": 0},
            "ToLabel": {},
        },
    },
}
default_toy_img_transform = edict()
for k, v in transform_dict.items():
    default_toy_img_transform[k] = {}
    for _k, _v in v.items():
        default_toy_img_transform[k][_k] = TransformInterface(_v)
