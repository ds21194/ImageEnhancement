import os
import random


def relpath(path):
    """
    Returns the relative path to the script's location

    :param path: a string representation of a path.
    :return:
    """
    return os.path.join(os.path.dirname(__file__), path)


def list_images(path, use_shuffle=True):
    """
    Returns a list of paths to images found at the specified directory.

    :param path: path to a directory to search for images.
    :param use_shuffle: option to shuffle order of files. Uses a fixed shuffled order.
    :return: list files of images inside path. will return only images with 'jpg' or 'png' format.
    """

    def is_image(filename):
        return os.path.splitext(filename)[-1][1:].lower() in ['jpg', 'png']

    # get list of all images with format 'png' or 'jpg'
    filtered_images = filter(is_image, os.listdir(path))

    # for every image add the relative path given to the image name
    images = list(map(lambda x: os.path.join(path, x), filtered_images))

    # Shuffle with a fixed seed without affecting global state
    if use_shuffle:
        s = random.getstate()
        random.seed()
        random.shuffle(images)
        random.setstate(s)
    return images


def images_for_denoising():
    """
    Returns a list of image paths to be used for image denoising
    """
    return list_images(relpath('datasets\\image_dataset\\train'), True)


def images_for_deblurring():
    """
    Returns a list of image paths to be used for text deblurring
    """
    return list_images(relpath('datasets\\text_dataset\\train'), True)


def images_for_super_resolution():
    """
    Returns a list of image paths to be used for image super-resolution
    """
    return list_images(relpath('image_dataset\\train'), True)

