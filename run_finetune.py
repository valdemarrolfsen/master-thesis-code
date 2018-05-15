import os

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam

from keras_utils.callbacks import callbacks, ValidationCallback
from keras_utils.generators import create_generator
from keras_utils.losses import binary_soft_jaccard_loss
from keras_utils.metrics import binary_jaccard_distance_rounded
from networks.unet.train import session_config
from networks.unet.unet import build_unet

datasets = [
    'buildings',
    'roads',
    'vegetation',
    'water']


def run():
    np.random.seed(2)
    tf.set_random_seed(2)

    base_lr = 0.0002
    max_lr = 0.002
    data_dir = '/data/{}/'
    logs_dir = 'logs/unet-{}-finalv2'
    weights_dir = 'weights_train'
    weights_name = 'unet-{}-finalv2'
    input_size = (320, 320)
    batch_size = 20
    binary = True
    session_config()
    for dataset in datasets:
        train_generator, num_samples = create_generator(
            os.path.join(data_dir.format(dataset), 'train'),
            input_size,
            batch_size,
            1,
            rescale=True,
            binary=binary,
            augment=False,
            mean=np.array([[[0.01279744, 0.01279744, 0.01279744]]]),
            std=np.array([[[0.11312577, 0.11312577, 0.11312577]]])
        )

        val_generator, val_samples = create_generator(
            os.path.join(data_dir.format(dataset), 'val'),
            input_size,
            batch_size,
            1,
            rescale=True,
            binary=binary,
            augment=False,
            mean=np.array([[[0.01279744, 0.01279744, 0.01279744]]]),
            std=np.array([[[0.11312577, 0.11312577, 0.11312577]]])
        )

        print('Running training for {}'.format(dataset))
        model = build_unet(input_size, nb_classes=1)
        model.summary()
        model.compile(
            optimizer=Adam(lr=max_lr),
            loss=binary_soft_jaccard_loss,
            metrics=['acc', binary_jaccard_distance_rounded])

        # weight = 'weights_train/weights.unet-{}-final.h5'.format(dataset)
        # print('Loading weights: {}'.format(weight))
        # model.load_weights(weight)

        steps_per_epoch = num_samples // batch_size
        cyclic = 'triangular2'

        cb = [ValidationCallback(val_samples // batch_size, val_generator)] + callbacks(
            logs_dir.format(dataset),
            filename=weights_name.format(dataset), weightsdir=weights_dir,
            monitor_val='mIOU',
            base_lr=base_lr, max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            cyclic=cyclic)
        model.fit_generator(
            generator=train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=100, verbose=True,
            workers=8,
            callbacks=cb
        )

        K.clear_session()


if __name__ == '__main__':
    run()
