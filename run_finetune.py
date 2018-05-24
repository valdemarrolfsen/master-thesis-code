import numpy as np
import os
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam

from keras_utils.callbacks import callbacks, ValidationCallback
from keras_utils.generators import create_generator
from keras_utils.losses import binary_soft_jaccard_loss, soft_jaccard_loss
from keras_utils.metrics import binary_jaccard_distance_rounded
from keras_utils.multigpu import get_number_of_gpus, ModelMGPU
from networks.densenet.densenet import build_densenet
from networks.unet.unet import build_unet

datasets = [
    'multiclass',
]

runs = [
    # {
    #     'name': 'unet-{}-final',
    #     'pre_weights_name': None,
    #     'network': 'unet',
    #     'base_lr': 0.0002,
    #     'max_lr': 0.002,
    #     'input_size': 320,
    #     'batch_size': 20,
    #     'rescale_masks': False,
    #     'datasets': ['multiclass']
    # },
    {
        'name': 'unet-{}-final-finetune',
        'pre_weights_name': 'unet-{}-final',
        'network': 'unet',
        'base_lr': 0.00002,
        'max_lr': 0.0002,
        'input_size': 512,
        'batch_size': 10,
        'rescale_masks': True,
        'datasets': ['multiclass']
    },
    {
        'name': 'unet-{}-finetune',
        'pre_weights_name': 'unet-{}-final',
        'network': 'unet',
        'base_lr': 0.00002,
        'max_lr': 0.0002,
        'input_size': 512,
        'batch_size': 10,
        'rescale_masks': False,
        'datasets': ['inra']
    }
]


def run():
    np.random.seed(2)
    tf.set_random_seed(2)
    data_dir = '/data/{}/'
    weights_dir = 'weights_train'

    for i, run in enumerate(runs):
        base_lr = run['base_lr']
        max_lr = run['max_lr']
        input_size = (run['input_size'], run['input_size'])
        weights_name = run['name']
        logs_dir = 'logs/{}'.format(run['name'])
        batch_size = run['batch_size']

        print("Running for config {}".format(run))

        for j, dataset in enumerate(run['datasets']):

            binary = True if dataset != 'multiclass' else False
            nb_classes = 1 if binary else 5

            print(binary)

            print('Running training for {}'.format(dataset))

            train_generator, num_samples = create_generator(
                os.path.join(data_dir.format(dataset), 'train'),
                input_size,
                batch_size,
                nb_classes=nb_classes,
                rescale=run['rescale_masks'],
                binary=binary,
                augment=False,
                mean=np.array([[[0.36654497, 0.35386439, 0.30782658]]]),
                std=np.array([[[0.19212837, 0.19031791, 0.18903286]]])
            )

            val_generator, val_samples = create_generator(
                os.path.join(data_dir.format(dataset), 'val'),
                input_size,
                batch_size,
                nb_classes=nb_classes,
                rescale=run['rescale_masks'],
                binary=binary,
                augment=False,
                mean=np.array([[[0.36654497, 0.35386439, 0.30782658]]]),
                std=np.array([[[0.19212837, 0.19031791, 0.18903286]]])
            )

            if run['network'] == 'unet':
                model = build_unet(input_size, nb_classes=nb_classes)
            else:
                model = build_densenet(input_size, nb_classes, 67)

            model.summary()
            gpus = get_number_of_gpus()
            print('Fund {} gpus'.format(gpus))
            if gpus > 1:
                model = ModelMGPU(model, gpus)

            if binary:
                loss = binary_soft_jaccard_loss
            else:
                loss = soft_jaccard_loss
            model.compile(
                optimizer=Adam(),
                loss=loss,
                metrics=['acc', binary_jaccard_distance_rounded])

            if run['pre_weights_name']:
                pre_weights_name = run['pre_weights_name'].format(dataset)
                weight = 'weights_train/weights.{}.h5'.format(pre_weights_name)
                print('Loading weights: {}'.format(weight))
                model.load_weights(weight)

            steps_per_epoch = num_samples // batch_size
            cyclic = 'triangular2'

            cb = [ValidationCallback(val_samples // batch_size, val_generator, binary=binary)] + callbacks(
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
