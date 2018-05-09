import os

import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam

from keras_utils.callbacks import callbacks
from keras_utils.losses import binary_soft_jaccard_loss
from keras_utils.metrics import binary_jaccard_distance_rounded
from keras_utils.multigpu import get_number_of_gpus, ModelMGPU
from networks.segcaps.capsnet import CapsNetR3
from networks.segcaps.data import create_caps_generator


def session_config():
    """
    Custom configs for tensorflow session, such as allowing growth for gpu mem.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    K.set_session(sess)  # set this TensorFlow session as the default session for Keras


def train_capsnet(data_dir, logdir, weights_dir, weights_name, input_size, nb_classes, batch_size, pre_trained_weight,
                   augment):
    session_config()
    model = CapsNetR3(input_size, n_class=nb_classes)[0]
    model.summary()
    gpus = get_number_of_gpus()
    print('Found {} gpus'.format(gpus))
    if gpus > 1:
        model = ModelMGPU(model, gpus)

    loss = binary_soft_jaccard_loss

    model.compile(
        optimizer=Adam(lr=1e-3),
        loss=loss,
        metrics=['acc', binary_jaccard_distance_rounded])

    train_generator, num_samples = create_caps_generator(os.path.join(data_dir, 'train'), input_size, batch_size, nb_classes, rescale=False, binary=True,
                                                    augment=False)
    val_generator, val_samples = create_caps_generator(os.path.join(data_dir, 'val'), input_size, batch_size, nb_classes, rescale=False, binary=True,
                                                  augment=False)

    if pre_trained_weight:
        print('Loading weights: {}'.format(pre_trained_weight))
        model.load_weights(pre_trained_weight)
    steps_per_epoch = num_samples // batch_size

    model.fit_generator(
        generator=train_generator,
        validation_data=val_generator,
        validation_steps=val_samples // batch_size,
        steps_per_epoch=1000,
        epochs=10000, verbose=True,
        workers=8,
        callbacks=callbacks(logdir, filename=weights_name, weightsdir=weights_dir, monitor_val='val_binary_jaccard_distance_rounded',
                            steps_per_epoch=steps_per_epoch),
        )
