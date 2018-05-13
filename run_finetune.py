import os

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam

from keras_utils.callbacks import callbacks
from keras_utils.generators import create_generator
from keras_utils.losses import binary_soft_jaccard_loss
from keras_utils.metrics import binary_jaccard_distance_rounded
from networks.unet.train import session_config
from networks.unet.unet import build_unet

datasets = [
    #'buildings',
    #'roads',
    'vegetation',
    'water']


def run():
    np.random.seed(2)
    tf.set_random_seed(2)

    base_lr = 0.00002
    max_lr = 0.0002
    data_dir = '/data/{}/'
    logs_dir = 'logs/unet-{}-final-finetune'
    weights_dir = 'weights_train'
    input_size = (512, 512)
    batch_size = 12

    binary = True
    session_config()
    for dataset in datasets:
        weights_name = 'unet-{}-final-finetune'
        train_generator, num_samples = create_generator(os.path.join(data_dir.format(dataset), 'train'), input_size, batch_size, nb_classes=1, rescale=False,
                                                        binary=binary,
                                                        augment=False)
        val_generator, val_samples = create_generator(os.path.join(data_dir.format(dataset), 'val'), input_size, batch_size, nb_classes=1, rescale=False,
                                                      binary=binary,
                                                      augment=False)

        print('Running finetuning for {}'.format(dataset))
        model = build_unet(input_size, nb_classes=1)
        model.summary()
        model.compile(
            optimizer=Adam(lr=max_lr),
            loss=binary_soft_jaccard_loss,
            metrics=['acc', binary_jaccard_distance_rounded])

        initial_epoch = 0
        if dataset == 'vegetation':
            print('Continue on vegetation')
            weight = 'weights_train/weights.unet-vegetation-final-finetune.h5'
            weights_name += '-continue'
            initial_epoch = 8
        else:
            weight = 'weights_train/weights.unet-{}-final.h5'.format(dataset)

        print('Loading weights: {}'.format(weight))
        model.load_weights(weight)

        steps_per_epoch = num_samples // batch_size
        cyclic = 'triangular2'
        model.fit_generator(
            generator=train_generator,
            validation_data=val_generator,
            validation_steps=val_samples // batch_size,
            steps_per_epoch=steps_per_epoch,
            epochs=10000, verbose=True,
            workers=8,
            initial_epoch=initial_epoch,
            callbacks=callbacks(logs_dir.format(dataset),
                                filename=weights_name.format(dataset), weightsdir=weights_dir,
                                monitor_val='val_binary_jaccard_distance_rounded',
                                base_lr=base_lr, max_lr=max_lr,
                                steps_per_epoch=steps_per_epoch,
                                cyclic=cyclic)
        )

        K.clear_session()


if __name__ == '__main__':
    run()
