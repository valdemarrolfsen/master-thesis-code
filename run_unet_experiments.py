import numpy as np
import tensorflow as tf
import os
from keras import backend as K
from keras.optimizers import Adam, Nadam

from keras_utils.callbacks import callbacks
from keras_utils.generators import create_generator
from keras_utils.losses import binary_soft_jaccard_loss
from keras_utils.metrics import binary_jaccard_distance_rounded
from networks.unet.train import session_config
from networks.unet.unet import build_unet

experiments = [
    {'optimizer': 'adam', 'dropout': 0.2, 'loss': 'binary_crossentropy', 'lr': 'annealing'},
    {'optimizer': 'adam', 'dropout': 0.2, 'loss': 'binary_crossentropy', 'lr': 'triangular2'},
    {'optimizer': 'adam', 'dropout': 0.2, 'loss': 'binary_crossentropy', 'lr': 'triangular'},
    {'optimizer': 'adam', 'dropout': 0.2, 'loss': 'jaccard', 'lr': 'annealing'},
    {'optimizer': 'adam', 'dropout': 0.2, 'loss': 'jaccard', 'lr': 'triangular2'},
    {'optimizer': 'adam', 'dropout': 0.2, 'loss': 'jaccard', 'lr': 'triangular'},

    {'optimizer': 'adam', 'dropout': 0.5, 'loss': 'binary_crossentropy', 'lr': 'annealing'},
    {'optimizer': 'adam', 'dropout': 0.5, 'loss': 'binary_crossentropy', 'lr': 'triangular2'},
    {'optimizer': 'adam', 'dropout': 0.5, 'loss': 'binary_crossentropy', 'lr': 'triangular'},
    {'optimizer': 'adam', 'dropout': 0.5, 'loss': 'jaccard', 'lr': 'annealing'},
    {'optimizer': 'adam', 'dropout': 0.5, 'loss': 'jaccard', 'lr': 'triangular2'},
    {'optimizer': 'adam', 'dropout': 0.5, 'loss': 'jaccard', 'lr': 'triangular'},

    {'optimizer': 'nadam', 'dropout': 0.2, 'loss': 'binary_crossentropy', 'lr': 'annealing'},
    {'optimizer': 'nadam', 'dropout': 0.2, 'loss': 'binary_crossentropy', 'lr': 'triangular2'},
    {'optimizer': 'nadam', 'dropout': 0.2, 'loss': 'binary_crossentropy', 'lr': 'triangular'},
    {'optimizer': 'nadam', 'dropout': 0.2, 'loss': 'jaccard', 'lr': 'annealing'},
    {'optimizer': 'nadam', 'dropout': 0.2, 'loss': 'jaccard', 'lr': 'triangular2'},
    {'optimizer': 'nadam', 'dropout': 0.2, 'loss': 'jaccard', 'lr': 'triangular'},

    {'optimizer': 'nadam', 'dropout': 0.5, 'loss': 'binary_crossentropy', 'lr': 'annealing'},
    {'optimizer': 'nadam', 'dropout': 0.5, 'loss': 'binary_crossentropy', 'lr': 'triangular2'},
    {'optimizer': 'nadam', 'dropout': 0.5, 'loss': 'binary_crossentropy', 'lr': 'triangular'},
    {'optimizer': 'nadam', 'dropout': 0.5, 'loss': 'jaccard', 'lr': 'annealing'},
    {'optimizer': 'nadam', 'dropout': 0.5, 'loss': 'jaccard', 'lr': 'triangular2'},
    {'optimizer': 'nadam', 'dropout': 0.5, 'loss': 'jaccard', 'lr': 'triangular'},
]


def get_optimizer(name, lr):
    if name == 'nadam':
        return Nadam(lr=lr)
    else:
        return Adam(lr=lr)


def get_loss(name):
    if name == 'binary_crossentropy':
        return name
    else:
        return binary_soft_jaccard_loss


def run():
    np.random.seed(2)
    tf.set_random_seed(2)

    base_lr = 0.0002
    max_lr = 0.002
    data_dir = '/data/buildings/'
    logs_dir = 'logs/{}'
    weights_dir = 'weights_train'
    weights_name = 'unet-experiment-{}'
    input_size = (320, 320)
    batch_size = 20
    start_from = 6
    binary = True
    session_config()

    train_generator, num_samples = create_generator(os.path.join(data_dir, 'train'), input_size, batch_size, nb_classes=1, rescale=False,
                                                    binary=binary,
                                                    augment=False)
    val_generator, val_samples = create_generator(os.path.join(data_dir, 'val'), input_size, batch_size, nb_classes=1, rescale=False,
                                                  binary=binary,
                                                  augment=False)

    for i, options in enumerate(experiments):
        if i < start_from:
            continue
        print('Running experiment {} with options: {}'.format(str(i), options))
        optimizer = get_optimizer(options['optimizer'], max_lr)
        loss = get_loss(options['loss'])
        dropout = options['dropout']

        model = build_unet(input_size, nb_classes=1, dropout=dropout)
        model.summary()
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['acc', binary_jaccard_distance_rounded])

        steps_per_epoch = num_samples // batch_size

        lr_opt = options['lr']
        if lr_opt == 'annealing':
            cyclic = None
        else:
            cyclic = lr_opt

        print('Experiment {} using lr: {}'.format(str(i), cyclic))
        model.fit_generator(
            generator=train_generator,
            validation_data=val_generator,
            validation_steps=val_samples // batch_size,
            steps_per_epoch=steps_per_epoch,
            epochs=20, verbose=True,
            workers=8,
            callbacks=callbacks(logs_dir.format(str(i)),
                                filename=weights_name.format(str(i)), weightsdir=weights_dir,
                                monitor_val='val_binary_jaccard_distance_rounded',
                                base_lr=base_lr, max_lr=max_lr,
                                steps_per_epoch=steps_per_epoch,
                                cyclic=cyclic
                                )
        )

        K.clear_session()


if __name__ == '__main__':
    run()
