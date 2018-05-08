import json

import numpy as np
import os
import tensorflow as tf
from keras.backend import set_session
from keras.optimizers import Adam

from keras_utils.callbacks import callbacks, CyclicLR
from keras_utils.generators import create_generator
from keras_utils.losses import binary_soft_jaccard_loss, soft_jaccard_loss, lovasz_hinge
from keras_utils.metrics import f1_score, binary_jaccard_distance_rounded
from keras_utils.multigpu import get_number_of_gpus, ModelMGPU
from networks.densenet.densenet import build_densenet

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


def lrtest_densenet(data_dir, logdir, weights_dir, weights_name, input_size, nb_classes, batch_size, config, initial_epoch, pre_trained_weight,
                   augment):
    session_config()
    model = build_densenet(input_size, nb_classes, config=config)

    gpus = get_number_of_gpus()
    print('Found {} gpus'.format(gpus))
    if gpus > 1:
        model = ModelMGPU(model, gpus)

    binary = nb_classes == 1
    if binary:
        loss = binary_soft_jaccard_loss
    else:
        loss = soft_jaccard_loss

    model.compile(
        optimizer=Adam(lr=1e-3),
        loss=loss,
        metrics=['acc', binary_jaccard_distance_rounded])

    train_generator, num_samples = create_generator(os.path.join(data_dir, 'train'), input_size, batch_size, nb_classes, rescale=False, binary=binary,
                                                    augment=augment)
    steps_per_epoch = num_samples // batch_size
    if augment:
        steps_per_epoch = steps_per_epoch * 4

    steps_per_epoch = 100

    clr = CyclicLR(base_lr=0, max_lr=1e-1, step_size=5 * steps_per_epoch, mode='triangular')
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=5, verbose=True,
        workers=8,
        callbacks=[clr])

    h = clr.history
    lr = h['lr']
    acc = h['acc']
    print(lr)
    print()
    print(acc)
