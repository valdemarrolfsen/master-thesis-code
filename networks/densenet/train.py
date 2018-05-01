import os
from keras.backend import set_session
import tensorflow as tf
from keras.optimizers import Adam

from keras_utils.callbacks import callbacks
from keras_utils.generators import create_generator
from keras_utils.losses import binary_soft_jaccard_loss, soft_jaccard_loss
from keras_utils.metrics import binary_jaccard_distance_rounded
from networks.densenet.densenet import build_densenet


def session_config():
    """
    Custom configs for tensorflow session, such as allowing growth for gpu mem.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras


def train_densenet(data_dir, logdir, weights_dir, weights_name, input_size, nb_classes, batch_size, config, initial_epoch, pre_trained_weight,
                   augment):
    session_config()
    model = build_densenet(input_size, nb_classes, config=config)

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
    val_generator, val_samples = create_generator(os.path.join(data_dir, 'val'), input_size, batch_size, nb_classes, rescale=False, binary=binary,
                                                  augment=augment)

    if pre_trained_weight:
        print('Loading weights: {}'.format(pre_trained_weight))
        model.load_weights(pre_trained_weight)

    steps_per_epoch = num_samples // batch_size
    if augment:
        print('Using augmentation')
        # calculate steps automatically.
        steps_per_epoch = 4 * steps_per_epoch

    model.fit_generator(
        generator=train_generator,
        validation_data=val_generator,
        validation_steps=val_samples // batch_size,
        steps_per_epoch=steps_per_epoch,
        epochs=10000, verbose=True,
        workers=8,
        callbacks=callbacks(logdir, filename=weights_name, weightsdir=weights_dir, monitor_val='val_acc', steps_per_epoch=steps_per_epoch),
        initial_epoch=initial_epoch)
