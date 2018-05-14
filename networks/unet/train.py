import numpy as np
import os
import tensorflow as tf
from keras.backend import set_session
from keras.optimizers import Adam

from keras_utils.callbacks import callbacks, ValidationCallback
from keras_utils.generators import create_generator
from keras_utils.losses import binary_soft_jaccard_loss, soft_jaccard_loss, lovasz_hinge
from keras_utils.metrics import f1_score, binary_jaccard_distance_rounded
from networks.unet.unet import build_unet, build_unet16

np.random.seed(2)
tf.set_random_seed(2)


def session_config():
    """
    Custom configs for tensorflow session, such as allowing growth for gpu mem.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras


def train_unet(data_dir, logdir, weights_dir, weights_name, input_size, nb_classes, batch_size, initial_epoch,
               pre_trained_weight, learning_rate, augment):
    session_config()
    model = build_unet(input_size, nb_classes)
    model.summary()
    binary = nb_classes == 1
    if binary:
        loss = binary_soft_jaccard_loss
    else:
        loss = soft_jaccard_loss

    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss=loss,
        metrics=['acc', binary_jaccard_distance_rounded])

    train_generator, num_samples = create_generator(
        os.path.join(data_dir, 'train'),
        input_size,
        batch_size,
        nb_classes,
        rescale=False,
        binary=binary,
        augment=augment)

    val_generator, val_samples = create_generator(
        os.path.join(data_dir, 'val'),
        input_size,
        batch_size,
        nb_classes,
        rescale=False,
        binary=binary,
        augment=augment)

    if pre_trained_weight:
        print('Loading weights: {}'.format(pre_trained_weight))
        model.load_weights(pre_trained_weight)
    steps_per_epoch = num_samples // batch_size

    if augment:
        steps_per_epoch = steps_per_epoch * 4

    cb = [ValidationCallback(val_samples // batch_size, val_generator)] + callbacks(
        logdir,
        filename=weights_name,
        weightsdir=weights_dir,
        monitor_val='mIOU',
        base_lr=0.0002,
        max_lr=0.002,
        steps_per_epoch=steps_per_epoch
    )
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=10000, verbose=True,
        workers=8,
        callbacks=cb,
        initial_epoch=initial_epoch)
