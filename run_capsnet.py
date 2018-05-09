import argparse

from networks.segcaps.train import train_capsnet


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
                        help='The name of the pretrained weight file',
                        default=''
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

    parser.add_argument('--initial-epoch',
                        type=int,
                        help='Initial epoch',
                        default=0
                        )

    parser.add_argument('--learning-rate',
                        type=float,
                        help='Learning rate',
                        default=1e-4
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

    train_capsnet(args.data_dir,
               args.logs_dir,
               args.weights_dir,
               args.weights_name,
               (args.input_size, args.input_size),
               args.classes,
               args.batch_size,
               args.pre_trained_weight,
               args.augment)
