import os

from keras_utils.callbacks import callbacks
from keras_utils.generators import create_generator
from networks.unet.unet import build_unet
import tensorflow as tf
import numpy as np
np.random.seed(2)
tf.set_random_seed(2)


def train_unet(data_dir, logdir, weights_dir, input_size, nb_classes, batch_size, initial_epoch):
    model = build_unet(nb_classes, input_size)
    train_generator, num_samples = create_generator(os.path.join(data_dir, 'train'), input_size, batch_size, nb_classes, rescale=False)
    val_generator, val_samples = create_generator(os.path.join(data_dir, 'val'), input_size, batch_size, nb_classes, rescale=False)

    model.fit_generator(
        generator=train_generator,
        validation_data=val_generator,
        validation_steps=val_samples//batch_size,
        steps_per_epoch=num_samples//batch_size,
        epochs=100, verbose=True,
        callbacks=callbacks(logdir, weightsdir=weights_dir), initial_epoch=initial_epoch)

