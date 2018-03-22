import os
from keras.optimizers import Adam

from keras_utils.callbacks import callbacks
from keras_utils.generators import create_generator
from networks.unet_binary.unet import build_unet_binary_deeper_elu, build_unet_binary_standard, jaccard_distance_loss, \
    jaccard_distance
import tensorflow as tf
import numpy as np
np.random.seed(2)
tf.set_random_seed(2)


def train_unet_binary(network, data_dir, logdir, weights_dir, weights_name, input_size, batch_size, initial_epoch):
    model = get_network(network, input_size)
    model.compile(
        optimizer=Adam(lr=1e-4),
        loss=jaccard_distance_loss,
        metrics=['binary_accuracy', jaccard_distance])

    train_generator, num_samples = create_generator(os.path.join(data_dir, 'train'), input_size, batch_size, 1, rescale=False, binary=True)
    val_generator, val_samples = create_generator(os.path.join(data_dir, 'val'), input_size, batch_size, 1, rescale=False, binary=True)

    model.fit_generator(
        generator=train_generator,
        validation_data=val_generator,
        validation_steps=val_samples//batch_size,
        steps_per_epoch=num_samples//batch_size,
        epochs=100, verbose=True,
        workers=8,
        callbacks=callbacks(logdir, filename=weights_name, weightsdir=weights_dir, monitor_val='val_jaccard_distance'),
        initial_epoch=initial_epoch)


def get_network(network, input_size):
    networks = {
        'standard': build_unet_binary_standard(input_size),
        'deeper-elu': build_unet_binary_deeper_elu(input_size)
    }
    return networks[network]
