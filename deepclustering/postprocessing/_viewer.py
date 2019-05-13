# this is the viewer script for 3D volumns visualization

# Define URL
# url = 'http://www.fil.ion.ucl.ac.uk/spm/download/data/attention/attention.zip'
#
# # Retrieve the data
# fn, info = urlretrieve(url, os.path.join(d, 'attention.zip'))
# import zipfile
#
# #
# # Extract the contents into the temporary directory we created earlier
# zipfile.ZipFile(fn).extractall(path=d)
import matplotlib.pyplot as plt
import numpy as np


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


from typing import List, Union
import torch

Tensor = Union[np.ndarray, torch.Tensor]


def multi_slice_viewer(img_volume: Tensor, gt_volumes: Union[Tensor, List[Tensor]] = None) -> None:
    if not isinstance(gt_volumes, list):
        gt_volumes = [gt_volumes]
    if gt_volumes[0] is not None:
        assert img_volume.shape == gt_volumes[0].shape
    fig, axs = plt.subplots(1, len(gt_volumes), )
    try:
        len(axs)
    except:
        axs = [axs]

    for i, (ax, volume) in enumerate(zip(axs, gt_volumes)):
        ax.gt_volume = volume
        ax.img_volume = img_volume
        ax.index = img_volume.shape[0] // 2
        ax.imshow(ax.img_volume[ax.index], cmap='gray')
        if volume is not None:
            ax.con = ax.contour(ax.gt_volume[ax.index])
        ax.set_title(f'plane = {ax.index}')
        ax.axis('off')

    remove_keymap_conflicts({'j', 'k'})

    fig.canvas.mpl_connect('key_press_event', process_key)
    fig.canvas.mpl_connect('scroll_event', process_mouse_wheel)


def process_mouse_wheel(event):
    fig = event.canvas.figure
    for i, ax in enumerate(fig.axes):
        if event.button == 'up':
            previous_slice(ax)
        elif event.button == 'down':
            next_slice(ax)
    fig.canvas.draw()


def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()


def previous_slice(ax):
    img_volume = ax.img_volume
    if ax.gt_volume is not None:
        for con in ax.con.collections:
            con.remove()
    ax.index = (ax.index - 1) if (ax.index - 1) >= 0 else 0  # wrap around using %
    ax.images[0].set_array(img_volume[ax.index])
    if ax.gt_volume is not None:
        ax.con = ax.contour(ax.gt_volume[ax.index])
    ax.set_title(f'plane = {ax.index}')


def next_slice(ax):
    img_volume = ax.img_volume
    if ax.gt_volume is not None:
        for con in ax.con.collections:
            con.remove()
    ax.index = (ax.index + 1) if (ax.index + 1) < img_volume.shape[0] else img_volume.shape[0] - 1
    ax.images[0].set_array(img_volume[ax.index])
    if ax.gt_volume is not None:
        ax.con = ax.contour(ax.gt_volume[ax.index])
    ax.set_title(f'plane = {ax.index}')


if __name__ == '__main__':
    from skimage import data

    astronaut = data.astronaut()
    ihc = data.immunohistochemistry()
    hubble = data.hubble_deep_field()

    # Initialize the subplot panels side by side

    import os

    # Create a temporary directory
    try:
        d = os.mkdir('file', )
    except FileExistsError:
        d = 'file'
    import os

    # Return the tail of the path
    os.path.basename('http://google.com/attention.zip')

    import nibabel

    # struct_arr = io.imread("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/attention-mri.tif")
    struct_arr = nibabel.load('file/attention/functional/snffM00587_0023.hdr').get_data()
    struct_arr = struct_arr.T

    struct_arr2 = nibabel.load('file/attention/functional/snffM00587_0030.hdr').get_data()
    struct_arr2 = struct_arr2.T

    multi_slice_viewer([struct_arr, struct_arr2])
    plt.show()
