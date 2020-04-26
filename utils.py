import os, random
import numpy as np
from skimage.draw import line
from imageio import imread
from skimage.color import rgb2gray

RGB_NUMBER = 2
GRAY_NUMBER = 1
COLOR_SIZE = 256


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
    :return:
    """

    def is_image(filename):
        return os.path.splitext(filename)[-1][1:].lower() in ['jpg', 'png']
    images = list(map(lambda x: os.path.join(path, x), filter(is_image, os.listdir(path))))
    # Shuffle with a fixed seed without affecting global state
    if use_shuffle:
        s = random.getstate()
        random.seed(1234)
        random.shuffle(images)
        random.setstate(s)
    return images


def images_for_denoising():
    """
    Returns a list of image paths to be used for image denoising
    """
    return list_images(relpath('datasets/image_dataset/train'), True)


def images_for_deblurring():
    """
    Returns a list of image paths to be used for text deblurring
    """
    return list_images(relpath('datasets/text_dataset/train'), True)


def images_for_super_resolution():
    """
    Returns a list of image paths to be used for image super-resolution
    """
    return list_images(relpath('image_dataset/train'), True)


def motion_blur_kernel(kernel_size, angle):
    """
    Returns a 2D image kernel for motion blur effect.

    :param kernel_size: the height and width of the kernel. Controls strength of blur.
    :param angle: angle in the range [0, np.pi) for the direction of the motion.
    :return:
    """
    if kernel_size % 2 == 0:
        raise ValueError('kernel_size must be an odd number!')
    if angle < 0 or angle > np.pi:
        raise ValueError('angle must be between 0 (including) and pi (not including)')
    norm_angle = 2.0 * angle / np.pi
    if norm_angle > 1:
        norm_angle = 1 - norm_angle
    half_size = kernel_size // 2
    if abs(norm_angle) == 1:
        p1 = (half_size, 0)
        p2 = (half_size, kernel_size-1)
    else:
        alpha = np.tan(np.pi * 0.5 * norm_angle)
        if abs(norm_angle) <= 0.5:
            p1 = (2*half_size, half_size - int(round(alpha * half_size)))
            p2 = (kernel_size-1 - p1[0], kernel_size-1 - p1[1])
        else:
            alpha = np.tan(np.pi * 0.5 * (1-norm_angle))
            p1 = (half_size - int(round(alpha * half_size)), 2*half_size)
            p2 = (kernel_size - 1 - p1[0], kernel_size-1 - p1[1])
    rr, cc = line(p1[0], p1[1], p2[0], p2[1])
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    kernel[rr, cc] = 1.0
    kernel /= kernel.sum()
    return kernel

