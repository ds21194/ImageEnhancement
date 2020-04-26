import tensorflow.keras as kr


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

