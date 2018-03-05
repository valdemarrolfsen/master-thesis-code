import os
from keras_utils.callbacks import callbacks
from keras_utils.generators import create_generator

from networks.pspnet import net_builder as layers


def train_psp(data_dir, logdir, input_shape, nb_classes, resnet_layers, batch_size, initial_epoch):
    model = layers.build_pspnet(nb_classes=nb_classes,
                                resnet_layers=resnet_layers,
                                input_shape=input_shape)

    train_generator, num_samples = create_generator(os.path.join(data_dir, 'train'), input_shape, batch_size, nb_classes)
    val_generator, val_samples = create_generator(os.path.join(data_dir, 'val'), input_shape, batch_size, nb_classes)
    model.fit_generator(
        generator=train_generator,
        validation_data=val_generator,
        validation_steps=val_samples,
        steps_per_epoch=num_samples // batch_size,
        epochs=100, verbose=True,
        callbacks=callbacks(logdir), initial_epoch=initial_epoch)
