from keras.models import load_model
from keras_utils.callbacks import callbacks
from keras_utils.generators import create_generator

from networks.pspnet import net_builder as layers


def train_psp(datadir, logdir, input_size, nb_classes, resnet_layers, batch_size, weights, initial_epoch, steps_per_epoch):
    if weights is not None:
        model = load_model(weights)
    else:
        model = layers.build_pspnet(nb_classes=nb_classes,
                                    resnet_layers=resnet_layers,
                                    input_shape=input_size)
    train_generator = create_generator(datadir, input_size, batch_size)
    model.fit_generator(
        generator=train_generator,
        epochs=100, verbose=True, steps_per_epoch=steps_per_epoch,
        callbacks=callbacks(logdir), initial_epoch=initial_epoch)


class PSPNet(object):
    """
    Pyramid Scene Parsing Network by Hengshuang Zhao et al 2017
    """

    def __init__(self, nb_classes, resnet_layers, input_shape):
        self.input_shape = input_shape
        self.model = layers.build_pspnet(nb_classes=nb_classes,
                                         resnet_layers=resnet_layers,
                                         input_shape=self.input_shape)
