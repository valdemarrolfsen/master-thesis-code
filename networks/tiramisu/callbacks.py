import math
from keras.callbacks import LearningRateScheduler

from keras_utils.callbacks import base_callbacks


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.00001
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def tiramisu_callbacks(logdir, weightsdir):
    callbacks = base_callbacks(logdir, weightsdir)
    lrate = LearningRateScheduler(step_decay)
    return callbacks


