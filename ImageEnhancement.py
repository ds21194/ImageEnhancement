import numpy as np
import tensorflow.keras as kr

from constants import BIAS_VAL


def restore_image(corrupted_image, base_model):
    """
    given a trained model "base_model", and noised / blurred image, will return an improved image.
    :param corrupted_image: numpy ndarray
    :param base_model: trained model
    :return: ndarray representing an image
    """
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
