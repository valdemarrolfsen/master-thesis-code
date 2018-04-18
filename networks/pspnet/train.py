import os
from keras_utils.callbacks import callbacks
from keras_utils.generators import create_generator

from networks.pspnet import net_builder as layers


def train_psp(data_dir, logdir, weights_dir, weights_name, input_size, nb_classes, batch_size, initial_epoch, pre_trained_weight):
    model = layers.build_pspnet(nb_classes=nb_classes, input_shape=input_size)
    binary = nb_classes == 1
    train_generator, num_samples = create_generator(os.path.join(data_dir, 'train'), input_size, batch_size, nb_classes, rescale=False, binary=binary)
    val_generator, val_samples = create_generator(os.path.join(data_dir, 'val'), input_size, batch_size, nb_classes, rescale=False, binary=binary)

    if pre_trained_weight:
        print('Loading weights: {}'.format(pre_trained_weight))
        model.load_weights(pre_trained_weight)

    model.fit_generator(
        generator=train_generator,
        validation_data=val_generator,
        validation_steps=val_samples//batch_size,
        steps_per_epoch=num_samples//batch_size,
        epochs=10000, verbose=True,
        workers=8,
        callbacks=callbacks(logdir, filename=weights_name, weightsdir=weights_dir, monitor_val='val_acc'),
        initial_epoch=initial_epoch)

