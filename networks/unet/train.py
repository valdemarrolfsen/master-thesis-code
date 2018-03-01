import os

from keras_utils.callbacks import callbacks
from keras_utils.generators import create_generator
from networks.unet.unet import build_unet


def train_unet(data_dir, logdir, input_size, nb_classes, batch_size, initial_epoch, steps_per_epoch):
    model = build_unet(nb_classes, input_size[0], input_size[1], 3)
    train_generator = create_generator(os.path.join(data_dir, 'train'), input_size, batch_size, nb_classes)
    val_generator = create_generator(os.path.join(data_dir, 'val'), input_size, batch_size, nb_classes)
    model.fit_generator(
        generator=train_generator,
        validation_data=val_generator,
        validation_steps=1,
        epochs=100, verbose=True, steps_per_epoch=steps_per_epoch,
        callbacks=callbacks(logdir), initial_epoch=initial_epoch)

