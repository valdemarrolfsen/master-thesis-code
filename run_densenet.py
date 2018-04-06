import argparse

from networks.densenet.train import train_densenet
from comet_ml import Experiment

experiment = Experiment(api_key="NQEkxvH9S8g4YfcfTaN8zKOj9")


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
    args = parser.parse_args()
    print('Using args: ', args)

    return args


if __name__ == '__main__':
    # Defining arguments
    args = define_args()
    train_densenet(args.data_dir,
                   args.logs_dir,
                   args.weights_dir,
                   args.weights_name,
                   (args.input_size, args.input_size),
                   args.classes,
                   args.batch_size,
                   initial_epoch=0,
                   config=args.config)
