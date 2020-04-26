import numpy as np
from imageio import imread
from constants import COLOR_SIZE, RGB_NUMBER
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve
import utils


def read_image(filename, representation):
    """
    return the image with values between [0,1] with the representation
    :param filename: image path
    :param representation: 1 for grayscale and 2 for rgb
    :return: ndarray representing an image
    """
    image = imread(filename)
    image = image.astype(np.float64)
    image /= (COLOR_SIZE-1)

    if representation == RGB_NUMBER:
        return image
    image_gray = rgb2gray(image)
    return image_gray


def round_and_clip(image, min_clip=0, max_clip=1, round_to=255):
    image = np.round(image * round_to) / round_to
    return np.clip(image, min_clip, max_clip)


def add_gaussian_noise(image, min_sigma=0, max_sigma=0.2):
    sigma = np.random.uniform(min_sigma, max_sigma)
    noise = np.random.normal(0, sigma, size=image.shape)
    noised_im = image + noise

    return round_and_clip(noised_im)


def add_motion_blur(image, kernel_size, angle):
    """
    add motion blur with kernel_size and angle
    :param image: a grayscale image with values in the [0,1] range of type float64
    :param kernel_size: an odd integer specifying the size of the kernel
    :param angle: angle to apply with the motion blue, in radians
    :return: a grayscale image blurred, values in the [0,1] range of type float64
    """
    kernel = utils.motion_blur_kernel(kernel_size, angle)
    motioned_img = convolve(image, kernel)
    return round_and_clip(motioned_img)


def random_motion_blur(image, list_of_kernel_sizes, min_angle=0, max_angle=np.pi):
    """
    add motion blur with random size and random angle
    :param image: a grayscale image with values in the [0,1] range of type float64
    :param list_of_kernel_sizes: a list of odd integers
    :param min_angle: minimum of angle to apply
    :param max_angle: maximum of angle to apply
    :return: image blurred
    """

    kernel_size = np.random.choice(np.array(list_of_kernel_sizes))
    angle = np.random.uniform(min_angle, max_angle)
    return add_motion_blur(image, kernel_size, angle)
