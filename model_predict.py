import os

import numpy as np
from scipy import misc
from skimage.util import view_as_blocks
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import shift

from saved_model import SavedModel


def predict(model, path):
    image = misc.imread(path)
    image = image[:256*(image.shape[0] / 256), :256*(image.shape[1] / 256)]
    image = np.fliplr(image)

    ax0, ax1 = np.meshgrid(
        np.arange(image.shape[1]),
        np.arange(image.shape[0])
    )

    blocks = view_as_blocks(image, (256, 256))
    ax0_blocks = view_as_blocks(ax0, (256, 256))
    ax1_blocks = view_as_blocks(ax1, (256, 256))
    blocks = blocks.reshape(-1, 256, 256)
    ax0_blocks = ax0_blocks.reshape(-1, 256, 256)
    ax1_blocks = ax1_blocks.reshape(-1, 256, 256)

    images = [
        image,
        shift(image, [0, 64]),
        shift(image, [0, -64]),
        shift(image, [-64, 0]),
        shift(image, [64, 0]),
        shift(image, [64, 64]),
        shift(image, [-64, 64]),
        shift(image, [64, -64]),
        shift(image, [-64, -64])
    ]

    overlay = np.zeros_like(image).astype(np.float32)
    for image in images:
        for block, ax0b, ax1b in zip(blocks, ax0_blocks, ax1_blocks):

            x, y = ax0b[128, 128], ax1b[128, 128]
            block = misc.imresize(block, (64, 64))
            block = block.reshape([1, 64, 64, 1])
            block = (block / 255.0) - .5
            features = np.array([[0, 5, y, x]])
            preds = model.predict(block, features)[0]
            overlay[y - 128: y + 128, x - 128: x + 128] += np.exp(preds[1])

    overlay = gaussian_filter(overlay, sigma=128)
    return overlay


def show_prediction_overlay(image, overlay):
    plt.imshow(image, vmin=0, vmax=255, cmap='gray', origin='lower')
    plt.imshow(overlay, vmin=overlay.min(), vmax=overlay.max(), alpha=.2)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    image_dir = 'images/full/'
    paths = [image_dir + path for path in os.listdir(image_dir)]
    model = SavedModel('checkpoints/model-0.meta', 'checkpoints/model-490000')
    for path in paths:
        output_dist = predict(model, path)
