import os
from keras_utils.callbacks import callbacks
from keras_utils.generators import create_generator

from networks.pspnet import net_builder as layers


def train_psp(data_dir, logdir, weights_dir, input_size, nb_classes, resnet_layers, batch_size, initial_epoch):
    model = layers.build_pspnet(nb_classes=nb_classes,
                                resnet_layers=resnet_layers,
                                input_shape=input_size)

    train_generator, num_samples = create_generator(os.path.join(data_dir, 'train'), input_size, batch_size, 1,
                                                    binary=False, rescale=False)
    val_generator, val_samples = create_generator(os.path.join(data_dir, 'val'), input_size, batch_size, 1, binary=False,
                                                  rescale=False)
    model.fit_generator(
        generator=train_generator,
        validation_data=val_generator,
        validation_steps=val_samples // batch_size,
        steps_per_epoch=num_samples // batch_size,
        epochs=1000, verbose=True,
        callbacks=callbacks(logdir, weightsdir=weights_dir, monitor_val='val_acc'), initial_epoch=initial_epoch)
