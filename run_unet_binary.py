import argparse
from networks.unet_binary.train import train_unet_binary


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

    parser.add_argument('--input-size',
                        type=int,
                        help='Input size for the images used (default 713)',
                        default=713
                        )

    parser.add_argument('--batch-size',
                        type=int,
                        help='Batch size',
                        default=2
                        )

    parser.add_argument('--steps-per-epoch',
                        type=int,
                        help='Steps per epoch',
                        default=50
                        )

    args = parser.parse_args()
    print('Using args: ', args)

    return args


if __name__ == '__main__':
    # Defining arguments
    args = define_args()

    train_unet_binary(args.data_dir,
                      args.logs_dir,
                      args.weights_dir,
                      (args.input_size, args.input_size),
                      args.batch_size,
                      0)