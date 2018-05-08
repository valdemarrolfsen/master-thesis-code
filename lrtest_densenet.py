import argparse

from networks.densenet.lrtest import lrtest_densenet
from networks.densenet.train import train_densenet


def define_args():
    """
    Defining the arguments for running the PSPNetwork
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',
                        type=str,
                        help='Directory for retrieving the input data (default data/output)',
                        default='data/output',
                        )

    parser.add_argument('--logs-dir',
                        type=str,
                        help='Directory for logs',
                        default='logs',
                        )

    parser.add_argument('--weights-dir',
                        type=str,
                        help='Directory for weights',
                        default='weights_train/',
                        )

    parser.add_argument('--weights-name',
                        type=str,
                        help='The name of the weight file',
                        default='{epoch:02d}-{loss:.2f}'
                        )

    parser.add_argument('--pre-trained-weight',
                        type=str,
                        help='Path to weight to load',
                        default=None
                        )

    parser.add_argument('--input-size',
                        type=int,
                        help='Input size for the images used (default 713)',
                        default=713
                        )
    parser.add_argument('--classes',
                        type=int,
                        help='Number of classes in the dataset (default 2)',
                        default=2
                        )

    parser.add_argument('--batch-size',
                        type=int,
                        help='Batch size',
                        default=2
                        )
    parser.add_argument('--config',
                        type=int,
                        help='The config for the network, choices are 56, 67 and 103',
                        default=67
                        )
    parser.add_argument('--initial-epoch',
                        type=int,
                        help='Initial epoch',
                        default=0
                        )
    parser.add_argument('--augment',
                        type=bool,
                        help='If to augment images or not.',
                        default=False
                        )
    args = parser.parse_args()
    print('Using args: ', args)

    return args


if __name__ == '__main__':
    # Defining arguments
    args = define_args()
    lrtest_densenet(args.data_dir,
                   args.logs_dir,
                   args.weights_dir,
                   args.weights_name,
                   (args.input_size, args.input_size),
                   args.classes,
                   args.batch_size,
                   args.config,
                   args.initial_epoch,
                   args.pre_trained_weight,
                   args.augment)
