import numpy as np
from scipy.ndimage.filters import convolve
from skimage.draw import line


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
    kernel = motion_blur_kernel(kernel_size, angle)
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

