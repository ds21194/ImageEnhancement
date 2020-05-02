import numpy as np
from imageio import imread
from constants import COLOR_SIZE, RGB_NUMBER
from skimage.color import rgb2gray
from constants import GRAY_NUMBER, BIAS_VAL


def get_image_from_cache(cache, file_path):
    """
    cache the images in a dictionary. if the image has been processed before we will bring it from cache instead
    of re-reading it from the hard disk. the images are represented in grayscale.
    :param cache: dictionary of images-path to ndarray which represent the image.
    :param file_path: file path of an image to read
    :return: ndarray representing image in grayscale
    """
    if file_path in cache:
        return cache[file_path]
    image = read_image(file_path, GRAY_NUMBER)
    cache[file_path] = image
    return image


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


def get_random_patch(image, crop_size, corruption_func):
    """
    get random cropped-image from image, corrupted and normal
    :param corruption_func: function that corrupt image
    :param image:
    :param crop_size:
    :return: tuple (normal, corrupted) of two images with crop_size image size
    """
    start_heights = np.random.randint(0, image.shape[0] - crop_size[0] - 1)
    start_width = np.random.randint(0, image.shape[1] - crop_size[1] - 1)
    end_heights = start_heights + crop_size[0]
    end_width = start_width + crop_size[1]
    # normal image
    normal = image[start_heights:end_heights, start_width:end_width].copy()

    # corrupted image:
    corrupted = corruption_func(image)
    corrupted = corrupted[start_heights:end_heights, start_width:end_width].copy()

    return corrupted, normal


def get_cropped_image(image, crop_size, corruption_func):
    """
    get corrupted image
    :param image:
    :param crop_size:
    :param corruption_func:
    :return: tuple of (normal, corrupted) images cropped with size of crop_size
    """
    bigger_patch_size = tuple(3 * crop_size[i] for i in range(len(crop_size)))
    # get random patch to corrupt (bigger then the original patch)
    larger_crop = get_random_patch(image, bigger_patch_size, corruption_func)[0]
    corrupted_patch, normal_patch = get_random_patch(larger_crop, crop_size, corruption_func)
    # return patch sampled from corrupted image
    return corrupted_patch, normal_patch


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """

    :param filenames:
    :param batch_size:
    :param corruption_func:
    :param crop_size:
    :return:
    """
    cache = {}
    while True:
        target_batch = np.zeros((batch_size,) + crop_size + (1,))
        source_batch = np.zeros((batch_size,) + crop_size + (1,))
        for i in range(batch_size):
            index = np.random.randint(0, len(filenames) - 1)
            random_im = get_image_from_cache(cache, filenames[index])
            target, source = get_cropped_image(random_im, crop_size, corruption_func)
            target = target - BIAS_VAL
            source = source - BIAS_VAL
            target_batch[i, :, :, 0] = target
            source_batch[i, :, :, 0] = source
        yield (source_batch, target_batch)
        source_batch = None
        target_batch = None


def get_random_train_validation_set(images, percent=0.8):
    images = np.array(images)
    indexes = np.arange(len(images))
    train_indexes = np.random.choice(indexes, int(np.round(len(images) * percent)))
    validation_indexes = np.delete(indexes, train_indexes)
    return train_indexes, validation_indexes
