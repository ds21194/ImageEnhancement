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
    :param image: ndarray represent a grayscale image
    :param crop_size: tuple
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

    return normal, corrupted


def get_cropped_image(image, crop_size, corruption_func):
    """
    get corrupted image
    :param image:
    :param crop_size:
    :param corruption_func:
    :return: tuple of (normal, corrupted) images cropped with size of crop_size
    """
    # To ensure randomality of the patch, I'm taking first bigger patch in random,
    # and out of that patch I'm creating the final image patch
    bigger_patch_size = tuple(3 * crop_size[i] for i in range(len(crop_size)))

    # get random patch to corrupt
    larger_crop = get_random_patch(image, bigger_patch_size, corruption_func)[0]

    normal_patch, corrupted_patch = get_random_patch(larger_crop, crop_size, corruption_func)

    # return patch sampled from corrupted image
    return normal_patch, corrupted_patch


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    Generator of tuple containing 2 batches of images: (original, corrupted)
    The images are random crops from the original image, in shape of crop_size
    :param filenames: numpy array of images' path
    :param batch_size: number of images to return each time in the same numpy array ('batch')
    :param corruption_func:
    :param crop_size:
    :return: yielding (corrupted_image, original_batch)
    """
    cache = {}
    while True:
        # create the shape of the return value. for example: (32, 16, 16, 1)
        original_batch = np.zeros((batch_size,) + crop_size + (1,))
        corrupted_batch = np.zeros((batch_size,) + crop_size + (1,))

        for i in range(batch_size):
            index = np.random.randint(0, len(filenames) - 1)
            random_im = get_image_from_cache(cache, filenames[index])

            # create tuple of (patch, corrupted_patch) from the original image
            patch, patch_corrupted = get_cropped_image(random_im, crop_size, corruption_func)

            # add bias
            patch = patch - BIAS_VAL
            patch_corrupted = patch_corrupted - BIAS_VAL

            # add the image to the batch
            original_batch[i, :, :, 0] = patch
            corrupted_batch[i, :, :, 0] = patch_corrupted

        yield corrupted_batch, original_batch
        corrupted_batch = None
        original_batch = None


def get_random_train_validation_set(images, percent=0.8):
    """
    return randomly selected group of train images and the complement group to be a validation set
    training size will be 'percent' of the original 'images' list
    :param images: numpy array of images' path
    :param percent: the percent of the size of the train group
    :return: the indexes of each group in the original list
    """

    indexes = np.arange(len(images))
    train_indexes = np.random.choice(indexes, int(np.round(len(images) * percent)))
    validation_indexes = np.delete(indexes, train_indexes)

    return train_indexes, validation_indexes
