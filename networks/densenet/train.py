import os

from keras_utils.callbacks import callbacks
from keras_utils.generators import create_generator
from networks.densenet.densenet import build_densenet


def train_densenet(data_dir, logdir, weights_dir, weights_name, use_pre_trained_weight, input_size, classes, batch_size, initial_epoch, config):
    model = build_densenet(input_size, classes, config=config)
    binary = classes == 1
    train_generator, num_samples = create_generator(os.path.join(data_dir, 'train'), input_size, batch_size, classes, rescale=False, binary=binary)
    val_generator, val_samples = create_generator(os.path.join(data_dir, 'val'), input_size, batch_size, classes, rescale=False, binary=binary)

    if use_pre_trained_weight is not None:
        model.load_weights(use_pre_trained_weight)

    model.fit_generator(
        generator=train_generator,
        validation_data=val_generator,
        validation_steps=val_samples//batch_size,
        steps_per_epoch=num_samples//batch_size,
        epochs=10000, verbose=True,
        workers=8,
        callbacks=callbacks(logdir, filename=weights_name, weightsdir=weights_dir, monitor_val='val_acc'),
        initial_epoch=initial_epoch)
