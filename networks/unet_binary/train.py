import os

from keras_utils.callbacks import callbacks
from keras_utils.generators import create_generator
from networks.unet_binary.unet import build_unet_binary
import tensorflow as tf
import numpy as np
np.random.seed(2)
tf.set_random_seed(2)


def train_unet_binary(data_dir, logdir, weights_dir, input_size, batch_size, initial_epoch):
    model = build_unet_binary(input_size)
    train_generator, num_samples = create_generator(os.path.join(data_dir, 'train'), input_size, batch_size, 1, rescale=False, binary=True)
    val_generator, val_samples = create_generator(os.path.join(data_dir, 'val'), input_size, batch_size, 1, rescale=False, binary=True)

    model.fit_generator(
        generator=train_generator,
        validation_data=val_generator,
        validation_steps=val_samples//batch_size,
        steps_per_epoch=num_samples//batch_size,
        epochs=10000, verbose=True,
        workers=8,
        callbacks=callbacks(logdir, weightsdir=weights_dir, monitor_val='val_jaccard_coef_int'),
        initial_epoch=initial_epoch)

