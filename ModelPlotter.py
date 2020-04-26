import numpy as np
import matplotlib.pyplot as plt

from ModelTrainer import *


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
