
import keras as kr
import numpy as np

import utils
from NnArchitecture import build_nn_model
from DataManipulator import random_motion_blur, add_gaussian_noise
from DataCreator import load_dataset, get_random_train_validation_set
from constants import DEBLUR_PATCH_SIZE, DENOISE_PATCH_SIZE


def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples):
    """
    build a general training model
    :param model: untrained nn model
    :param images: list of images' path
    :param corruption_func:
    :param batch_size: number of images to learn in 'one act'
    :param steps_per_epoch:
    :param num_epochs:
    :param num_valid_samples:
    :return:
    """

    # compile the model with his optimizer and loss function.
    adam_opt = kr.optimizers.Adam(beta_2=0.9)
    kr.models.Model.compile(
        model,
        adam_opt,
        metrics=['accuracy'],
        loss='mean_squared_error')

    images = np.array(images)
    np.random.shuffle(images)  # TODO: consider removing it. redundant ? shuffled already when created

    train_indexes, validation_indexes = get_random_train_validation_set(images)

    train_gen = load_dataset(
        images[train_indexes],
        batch_size,
        corruption_func,
        model.input_shape[1:3])

    validation_gen = load_dataset(
        images[validation_indexes],
        batch_size,
        corruption_func,
        model.input_shape[1:3])

    history = kr.models.Model.fit_generator(
        model,
        train_gen,
        steps_per_epoch,
        num_epochs,
        verbose=1,
        validation_data=validation_gen,
        validation_steps=(num_valid_samples // batch_size))

    return history


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    """
    model learning for fixing noised images. quick_mode is mainly for debugging purpose, so the result will be faster.
    :param num_res_blocks: number of residual blocks inside the neural network architecture
    :param quick_mode: boolean parameter
    :return: trained model for noised images
    """
    # get list of images path:
    data = utils.images_for_denoising()

    denoise_model = build_nn_model(
        DENOISE_PATCH_SIZE['height'],
        DENOISE_PATCH_SIZE['width'], 48, num_res_blocks)
    # global training_history

    if quick_mode:
        train_model(denoise_model, data, lambda image: add_gaussian_noise(image, 0, 0.2),
                                       10, 3, 2, 30)
    else:
        train_model(denoise_model, data, lambda image: add_gaussian_noise(image, 0, 0.2),
                                       100, 100, 5, 1000)
    return denoise_model


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    """
    model learning for fixing blurred images. quick_mode is mainly for debugging purpose, so the result will be faster.
    :param num_res_blocks: number of residual blocks inside the neural network architecture
    :param quick_mode: boolean parameter
    :return: trained model for blurred images
    """
    # get list of images path:
    data = utils.images_for_deblurring()

    # define a deblurring model:
    deblur_model = build_nn_model(
        DEBLUR_PATCH_SIZE['height'],
        DEBLUR_PATCH_SIZE['width'], 32, num_res_blocks)

    if quick_mode:
        train_model(deblur_model, data, lambda image: random_motion_blur(image, [7]),
                    10, 3, 2, 30)
    else:
        train_model(deblur_model, data, lambda image: random_motion_blur(image, [7]),
                    100, 100, 10, 1000)

    return deblur_model

