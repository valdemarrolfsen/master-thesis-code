import argparse
import os
from networks.capsnet.capsulenet import train, CapsNet


if __name__ == "__main__":

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--data-dir', default='data/output', type=str)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr-decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam-recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift-fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--output-dir', default='./result')
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    args = parser.parse_args()

    print("Using args: {}".format(args))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # define model
    model, eval_model, manipulate_model = CapsNet(input_shape=(400, 400, 3),
                                                  n_class=2,
                                                  routings=args.routings)
    model.summary()

    train(model=model, args=args)
