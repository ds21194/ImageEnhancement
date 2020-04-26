import tensorflow.keras as kr
import numpy as np

import utils
import NnArchitecture as nnArchit
from DataManipulator import random_motion_blur, add_gaussian_noise
from DataCreator import load_dataset, get_random_train_validation_set


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


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    data = utils.images_for_denoising()

    denoise_model = nnArchit.build_nn_model(24, 24, 48, num_res_blocks)
    global training_history
    if quick_mode:
        training_history = nnArchit.train_model(denoise_model, data, lambda image: add_gaussian_noise(image, 0, 0.2),
                                       10, 3, 2, 30)
    else:
        training_history = nnArchit.train_model(denoise_model, data, lambda image: add_gaussian_noise(image, 0, 0.2),
                                       100, 100, 5, 1000)
    return denoise_model


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    data = utils.images_for_deblurring()
    deblur_model = nnArchit.build_nn_model(16, 16, 32, num_res_blocks)
    if quick_mode:
        nnArchit.train_model(deblur_model, data, lambda image: random_motion_blur(image, [7]),
                    10, 3, 2, 30)
    else:
        nnArchit.train_model(deblur_model, data, lambda image: random_motion_blur(image, [7]),
                    100, 100, 10, 1000)

    return deblur_model

