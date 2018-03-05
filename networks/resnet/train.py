import numpy as np
import os
import tensorflow as tf

from keras_utils.callbacks import base_callbacks
from keras_utils.generators import create_generator
from networks.resnet.resnet import build_atrous_resnet

np.random.seed(2)
tf.set_random_seed(2)


def train_resnet(data_dir, logdir, weightsdir, input_size, nb_classes, batch_size, initial_epoch=0):
    model = build_atrous_resnet(nb_classes, input_size)
    train_generator, num_samples = create_generator(os.path.join(data_dir, 'train'), input_size, batch_size, nb_classes)
    val_generator, val_samples = create_generator(os.path.join(data_dir, 'val'), input_size, batch_size, nb_classes)
    model.fit_generator(
        generator=train_generator,
        validation_data=val_generator,
        validation_steps=val_samples,
        steps_per_epoch=num_samples//batch_size,
        epochs=100, verbose=True,
        callbacks=base_callbacks(logdir, weightsdir=weightsdir), initial_epoch=initial_epoch)

