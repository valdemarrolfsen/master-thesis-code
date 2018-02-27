import argparse

from networks.pspnet.train import train_psp
from networks.capsnet.capsulenet import train, CapsNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Main menu for project")
    parser.add_argument(
        '-c', '--command',
        type=int,
        help="The command that should be run."
    )
    
    parser.add_argument(
        '--input-size',
        default=713,
        type=int,
        help='Input size of the images'
    )

    args = parser.parse_args()

    if args.command == 1:
        train_psp('data/output/', 'training', (args.input_size, args.input_size), 3, 50, 2, None, 0, ').')
    elif args.command == 2:
        print("Training Caps Net")

        model, eval_model, manipulate_model = CapsNet(input_shape=(args.input_size, args.input_size), n_class=3, routings=3)
        model.summary()

        if args.weights is not None:  # init the model weights with provided one
            model.load_weights(args.weights)

        if not args.testing:
            train(model=model, args=args)
        else:  # as long as weights are given, will run testing
            if args.weights is None:
                print('No weights are provided. Will test using random initialized weights.')
            manipulate_latent(manipulate_model, (x_test, y_test), args)
            test(model=eval_model, data=(x_test, y_test), args=args)
