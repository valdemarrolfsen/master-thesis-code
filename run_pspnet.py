import argparse
from networks.pspnet.train import train_psp


def define_args():
    """
    Defining the arguments for running the PSPNetwork
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
        type=str,
        help='Directory for retrieving the input data (default data/output)',
        default='data/output'
    )
    parser.add_argument('--logs_dir',
        type=str,
        help='Directory for storing processing logs (default logs/)',
        default='logs/'
    )
    parser.add_argument('--input-size',
        type=int,
        help='Input size for the images used (default 713)',
        default=713
    )
    parser.add_argument('--classes',
        type=float,
        help='Number of classes in the dataset (default 2)',
        default=2
    )
    parser.add_argument('--weights',
        type=int,
        help='Directory for storing weights (default: weights/)',
        default='weights/'
    )

    args = parser.parse_args()
    print('Using args: ', args)

    return args


if __name__ == '__main__':

    # Defining arguments
    args = define_args()

    train_psp(
        args.datadir,
        args.logdir,
        (args.input_size, args.input_size),
        args.classes,
        50,
        args.batch_size,
        args.weights,
        0,
        ').'
    )