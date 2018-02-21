import argparse
import os

from keras.models import load_model
from python_utils.callbacks import callbacks

import net_builder as layers
from utils import create_generator


def train(datadir, logdir, input_size, nb_classes, resnet_layers, batchsize, weights, initial_epoch, sep):
    if args.weights:
        model = load_model(weights)
    else:
        model = layers.build_pspnet(nb_classes=nb_classes,
                                    resnet_layers=resnet_layers,
                                    input_shape=input_size)
    train_generator = create_generator(datadir)
    model.fit_generator(
        generator=train_generator,
        epochs=100, verbose=True, steps_per_epoch=50,
        callbacks=callbacks(logdir), initial_epoch=initial_epoch)


class PSPNet(object):
    """Pyramid Scene Parsing Network by Hengshuang Zhao et al 2017"""

    def __init__(self, nb_classes, resnet_layers, input_shape):
        self.input_shape = input_shape
        self.model = layers.build_pspnet(nb_classes=nb_classes,
                                         resnet_layers=resnet_layers,
                                         input_shape=self.input_shape)
        # No weights for our purpose
        # print("Load pre-trained weights")
        # self.model.load_weights("weights/keras/pspnet101_voc2012.h5")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', type=int, default=473)
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--resnet_layers', type=int, default=50)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--initial_epoch', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--sep', default=').')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    train(args.datadir, args.logdir, (713, 713), args.classes, args.resnet_layers,
          args.batch, args.weights, args.initial_epoch, args.sep)
