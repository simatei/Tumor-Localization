from collections import defaultdict
from multiprocessing import Pool, cpu_count
import random
import json
import os

import numpy as np
import scipy.misc
import scipy.signal
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _assert_reconstruction(image, image_bytes):
    decoded = np.fromstring(image_bytes, dtype=np.uint8)
    decoded = decoded.reshape(64, 64)
    assert (decoded == image).all(), 'byte mismatch'


def make_example(image_sample, features, label):
    image_sample = image_sample.astype(np.uint8)
    image_sample_bytes = image_sample.tobytes()
    _assert_reconstruction(image_sample, image_sample_bytes)
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'label': _int64_feature(label),
                'image_raw': _bytes_feature(image_sample_bytes),
                'features': _int64_feature(features)
            }
        )
    )
    return example


def mask_from_boundary(metadata, full_image_dir):
    image_fname = metadata['IMAGE_PATH']
    image = Image.open(os.path.join(full_image_dir, image_fname))
    mask = np.zeros_like(np.array(image))

    boundary = metadata['ABNORMALITIES'][0]['BOUNDARIES'][0]
    boundary_mask = np.zeros_like(np.array(image))
    x, y = boundary[0: 2]

    x_to_y = defaultdict(list)
    for direction in boundary[2:]:
        if direction in {3, 4, 5}:
            y += 1
        elif direction in {0, 1, 7}:
            y -= 1
        if direction in {1, 2, 3}:
            x += 1
        elif direction in {5, 6, 7}:
            x -= 1
        x_to_y[x].append(y)
        boundary_mask[y-10: y + 10, x-10: x+10] = 1

    for x, ys in x_to_y.items():
        min_y = min(ys)
        max_y = max(ys)

        min_y = max(0, min(mask.shape[0] - 1, min_y))
        max_y = max(0, min(mask.shape[0] - 1, max_y))
        x = max(0, min(mask.shape[1] - 1, x))
        mask[min_y: max_y + 1, x] = 1

    return mask


def show_image(image):
    plt.imshow(image, vmin=0, vmax=255, cmap='gray', origin='lower')
    plt.axis('off')
    plt.show()


def add_image_to_axis(image, ax):
    ax.imshow(image, vmin=0, vmax=255, cmap='gray', origin='lower')
    ax.axis('off')
    return ax


def segment_mask(image):
    size = 16
    kernel = np.ones((size, size)) / size**2
    r = scipy.signal.convolve2d(image, kernel, mode='same')
    return (r > 65).astype(np.int) * 255


def show_segmentation(image):
    mask = segment_mask(image)
    plt.axis('off')
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1 = add_image_to_axis(image, ax1)
    ax2 = add_image_to_axis(mask, ax2)
    plt.show()


def show_samples(image, centers, num):
    centers = centers[:num] + centers[-num:]
    mask = np.zeros_like(image)

    border_size = 10
    inner = 128 - border_size

    for y, x in centers:
        single_mask = np.zeros_like(image)
        single_mask[y-128: y + 128, x-128: x+128] = 1
        single_mask[y-inner: y + inner, x-inner: x+inner] = 0
        mask = mask | single_mask

    mask = mask*255
    mask = np.ma.masked_where(mask < .5, mask)
    fig, ax = plt.subplots()
    ax.imshow(image, vmin=0, vmax=255, cmap='gray', origin='lower')
    ax.imshow(mask, vmin=0, vmax=255, cmap='rainbow', origin='lower')
    plt.axis('off')
    plt.show()


def subplot_samples(samples, num):
    samples = samples[:num/2] + samples[-num/2:]
    fig, axes = plt.subplots(num/2, 2)
    for sample, ax in zip(samples[:num], axes.ravel(order='F')):
        ax = add_image_to_axis(sample, ax)
        ax.set_aspect('equal')
    plt.subplots_adjust(wspace=.1, hspace=.1)
    plt.show()


def generate_samples(mask, image, metadata, write_fname):
    ys, xs = np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]))

    mask_xs, mask_ys = xs[mask.T == 1], ys[mask.T == 1]
    mask_coords = zip(mask_ys, mask_xs)
    min_x, max_x = mask_xs.min(), mask_xs.max()
    min_y, max_y = mask_ys.min(), mask_ys.max()

    lesion = metadata['ABNORMALITIES'][0]['LESION_TYPE']
    lesion = 1 if lesion == 'CALCIFICATION' else 0
    subtlety = int(metadata['ABNORMALITIES'][0]['SUBTLETY'])

    with tf.python_io.TFRecordWriter(write_fname) as writer:

        # generate 30 positive samples
        num_positives = 0
        while num_positives < 30:
            sampled_y, sampled_x = random.choice(mask_coords)
            perturbed_y = sampled_y + int(random.uniform(-56, 56))
            perturbed_x = sampled_x + int(random.uniform(-56, 56))

            perturbed_y = max(128, min(image.shape[0] - 128, perturbed_y))
            perturbed_x = max(128, min(image.shape[1] - 128, perturbed_x))

            sampled_center = (perturbed_y, perturbed_x)
            image_sample = image[
                sampled_center[0] - 128: sampled_center[0] + 128,
                sampled_center[1] - 128: sampled_center[1] + 128
            ]

            features = [lesion, subtlety, sampled_center[0], sampled_center[1]]
            image_sample = np.array(
                scipy.misc.imresize(image_sample, (64, 64)))
            example = make_example(image_sample, features, label=1)
            writer.write(example.SerializeToString())
            num_positives += 1

        # generate 30 negative samples
        num_negatives = 0
        while num_negatives < 30:

            # generate random patch from image
            y_range = (128, mask.shape[0] - 128)
            x_range = (128, mask.shape[1] - 128)

            sample_y = int(random.uniform(*y_range))
            sample_x = int(random.uniform(*x_range))

            sample_y = max(128, min(image.shape[0] - 128, sample_y))
            sample_x = max(128, min(image.shape[1] - 128, sample_x))

            # discard if it is actually a positive sample
            if (min_x - 128 < sample_x < max_x + 128) or \
               (min_y - 128 < sample_y < max_y + 128):
                continue

            sampled_center = (sample_y, sample_x)
            image_sample = image[
                sampled_center[0] - 128: sampled_center[0] + 128,
                sampled_center[1] - 128: sampled_center[1] + 128
            ]

            # downsample empty portion of image
            if image_sample[image_sample > 125].sum() < 200:
                if np.random.rand() > .01:
                    continue

            features = [lesion, subtlety, sampled_center[0], sampled_center[1]]
            image_sample = np.array(
                scipy.misc.imresize(image_sample, (64, 64)))
            example = make_example(image_sample, features, label=0)
            writer.write(example.SerializeToString())
            num_negatives += 1


def to_tfrecords(metadata):
    full_image_dir = 'images/full/'
    for i, image_metadata in enumerate(metadata):
        cc_metadata = image_metadata['CC_OVERLAY']
        mlo_metadata = image_metadata['MLO_OVERLAY']
        for view_metadata in [cc_metadata, mlo_metadata]:
            image_fname = view_metadata['IMAGE_PATH']
            write_fname = image_fname.replace('.jpg', '.tfrecords')

            write_fname = 'images/shards/' + write_fname
            if os.path.isfile(write_fname):
                continue

            image = Image.open(os.path.join(full_image_dir, image_fname))
            image_array = np.array(image)

            mask = mask_from_boundary(view_metadata, full_image_dir)
            generate_samples(mask, image_array, view_metadata, write_fname)


if __name__ == '__main__':
    with open('metadata/metadata.json', 'r') as json_file:
        metadata = json.load(json_file)

        partitions = []
        partition_size = 10
        for i in range(0, len(metadata), partition_size):
            partitions.append(metadata[i: i + partition_size])

        pool = Pool(processes=(cpu_count() - 1))
        pool.map(to_tfrecords, partitions)
