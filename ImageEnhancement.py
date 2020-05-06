import numpy as np
import keras as kr

from constants import BIAS_VAL


def load_trained_model(path):
    """
    Load a pre trained model. 'base_model' parameter for 'restore_image' function
    :param path: path to the model on the computer
    :return: a trained model
    """
    return kr.models.load_model(path)


def copy_restored_patch(model, origin, patch_shape, i, j):
    patch = origin[i:i + patch_shape[0], j:j + patch_shape[1]]
    patch = patch[..., np.newaxis]
    res = model.predict(patch[np.newaxis, ...])[0].copy()
    return res.reshape(res.shape[0:2])


def restore_image_v2(corrupted_image, base_model):
    im_shape = corrupted_image.shape
    patch_shape = base_model.input_shape[1:3]
    restored_image = np.zeros(im_shape)
    for i in range(0, im_shape[0]-patch_shape[0], patch_shape[0]):
        for j in range(0, im_shape[1]-patch_shape[1], patch_shape[1]):
            restored_image[i:i+patch_shape[0], j:j+patch_shape[1]] = copy_restored_patch(
                base_model,
                corrupted_image,
                patch_shape,
                i,
                j)

    i = im_shape[0] - patch_shape[0]
    for j in range(0, im_shape[1] - patch_shape[1], patch_shape[1]):
        restored_image[i:i + patch_shape[0], j:j + patch_shape[1]] = copy_restored_patch(
            base_model,
            corrupted_image,
            patch_shape,
            i,
            j)

    j = im_shape[1] - patch_shape[1]
    for i in range(0, im_shape[0] - patch_shape[0], patch_shape[0]):
        restored_image[i:i + patch_shape[0], j:j + patch_shape[1]] = copy_restored_patch(
            base_model,
            corrupted_image,
            patch_shape,
            i,
            j)

    j = im_shape[1] - patch_shape[1]
    i = im_shape[0] - patch_shape[0]
    restored_image[i:i + patch_shape[0], j:j + patch_shape[1]] = copy_restored_patch(
        base_model,
        corrupted_image,
        patch_shape,
        i,
        j)

    return restored_image


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

