import numpy as np
import tensorflow.keras as kr
import utils
from scipy.ndimage.filters import convolve
from random import uniform
import matplotlib.pyplot as plt
from imageio import imread
from skimage.color import rgb2gray

RGB_NUMBER = 2
GRAY_NUMBER = 1
COLOR_SIZE = 256

# ----------------------------------------------------------------------------------------------------- #

BIAS_VAL = 0.5


# -------------------------------------- Image Data Manipulation -------------------------------------- #


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


# -------------------------------------- Image Data Generator -------------------------------------- #


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


# -------------------------------------- Network Architecture -------------------------------------- #


def resblock(input_tensor, num_channels, kernel_size=(3, 3)):
    """
    residual block in ResNet architecture
    :param kernel_size:
    :param input_tensor:
    :param num_channels: number of channel on the result convolution
    :param kernel_size: size of the convolution to apply on the layer "Conv2D"
    :return: output tensor
    """
    conv1 = kr.layers.Conv2D(num_channels, kernel_size, padding='same')(input_tensor)
    act1 = kr.layers.Activation('relu')(conv1)
    conv2 = kr.layers.Conv2D(num_channels, kernel_size, padding='same')(conv1)
    add_layer = kr.layers.Add()([conv2, input_tensor])
    result = kr.layers.Activation('relu')(add_layer)
    return result


def build_nn_model(height, width, num_channels, num_res_blocks, kernel_size=(3, 3)):
    """

    :param height:
    :param width:
    :param num_channels:
    :param num_res_blocks:
    :param kernel_size:
    :return:
    """
    # first layer:
    input_layer = kr.layers.Input((height, width, 1))
    first_layer = kr.layers.Conv2D(num_channels, kernel_size, padding='same')(input_layer)
    act1 = kr.layers.Activation('relu')(first_layer)
    resblock_output = act1
    # other block-layers:
    for i in range(num_res_blocks):
        resblock_output = resblock(resblock_output, num_channels, kernel_size=kernel_size)
    # last layer om the model:
    last_conv_layer = kr.layers.Conv2D(1, kernel_size, padding='same')(resblock_output)
    add_layer = kr.layers.Add()([last_conv_layer, input_layer])

    return kr.models.Model(inputs=[input_layer], outputs=[add_layer])


def get_random_train_validation_set(images, percent=0.8):
    images = np.array(images)
    indexes = np.arange(len(images))
    train_indexes = np.random.choice(indexes, int(np.round(len(images) * percent)))
    validation_indexes = np.delete(indexes, train_indexes)
    return train_indexes, validation_indexes


def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples):
    # compile the model with his optimizer and loss function.
    adam_opt = kr.optimizers.Adam(beta_2=0.9)
    kr.Model.compile(model, adam_opt, loss='mean_squared_error')

    images = np.array(images)
    np.random.shuffle(images)

    train_indexes, val_indexes = get_random_train_validation_set(images)

    train_gen = load_dataset(images[train_indexes], batch_size, corruption_func, model.input_shape[1:3])
    validation_gen = load_dataset(images[val_indexes], batch_size, corruption_func, model.input_shape[1:3])

    history = kr.Model.fit_generator(model, train_gen, steps_per_epoch, num_epochs, verbose=1,
                                     validation_data=validation_gen, validation_steps=(num_valid_samples // batch_size))
    return history


# -------------------------------------- Image Restoration of Complete Images -------------------------------------- #


def restore_image(corrupted_image, base_model):
    corrupted_image = corrupted_image[..., np.newaxis]
    image_shape = corrupted_image.shape
    a = kr.layers.Input(shape=image_shape)
    b = base_model(a)
    new_model = kr.Model(inputs=a, outputs=b)

    corrupted_image = corrupted_image - BIAS_VAL
    clean_image = new_model.predict(corrupted_image[np.newaxis, ...])[0]
    clean_image = clean_image + BIAS_VAL
    clean_image = np.clip(clean_image, 0, 1)
    return clean_image[:, :, 0]


# -------------------------------------- Training Models: -------------------------------------- #


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    data = utils.images_for_denoising()

    denoise_model = build_nn_model(24, 24, 48, num_res_blocks)
    global training_history
    if quick_mode:
        training_history = train_model(denoise_model, data, lambda image: add_gaussian_noise(image, 0, 0.2),
                                       10, 3, 2, 30)
    else:
        training_history = train_model(denoise_model, data, lambda image: add_gaussian_noise(image, 0, 0.2),
                                       100, 100, 5, 1000)
    return denoise_model


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    data = utils.images_for_deblurring()
    deblur_model = build_nn_model(16, 16, 32, num_res_blocks)
    if quick_mode:
        train_model(deblur_model, data, lambda image: random_motion_blur(image, [7]),
                    10, 3, 2, 30)
    else:
        train_model(deblur_model, data, lambda image: random_motion_blur(image, [7]),
                    100, 100, 10, 1000)
    return deblur_model


def depth_effect_plot_for(learn_model_func, min_res_block_num=1, max_res_block_num=6,
                          show_plot=False, save_plot=False, describer="", quick_mode=False):
    loss_result = []
    axis_x = np.arange(min_res_block_num, max_res_block_num)
    for block_num in range(min_res_block_num, max_res_block_num):
        model = learn_model_func(block_num, quick_mode)
        history = model.history
        loss_result.append(history.history['val_loss'][-1])

    plt.clf()
    plt.plot(axis_x, loss_result)

    if save_plot:
        describer += ".png"
        plt.savefig(describer)
    if show_plot:
        plt.show()


def depth_effect_plot():
    depth_effect_plot_for(learn_denoising_model)
    depth_effect_plot_for(learn_deblurring_model)
