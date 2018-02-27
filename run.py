from networks.pspnet.train import train_psp
from networks.capsnet.capsulenet import train, CapsNet

choices = {
    1: "Generate new examples from dataset",
    2: "Train a network"
}

available_networks = {
    1: "PSPNet",
    2: "CapsNet"
}


def print_menu():
    global available_networks
    global choices

    print("========================================")
    print("       WELCOME TO MASTER THESIS")
    print("========================================\n")
    print("The choices avaliable are:\n")

    for k in choices:
        print("{}. {}".format(k, choices[k]))
    print()

    choice = int(input("Give a choice: "))

    if choice == 1:
        print("\nYou have chosen to generate a new set of training examples")
    elif choice == 2:
        print("You have chosen to train a network. The available networks are:\n")

        for k in available_networks:
            print("{}. {}".format(k, available_networks[k]))
        print()

        network = int((input("Choose a network: ")))

        print("You have chosen to train the {}\n".format(available_networks[network]))

        args = set_up_hyperparams(network)

        train_network(network, args)


def set_up_hyperparams(network):
    print("Choose the hyperparameters for this section:\n")

    args = {
        "datadir": input(
            "1. Indicate in which folder the program should locate the data: (data/output) ") or "data/output",
        "logdir": input("2. Indicate where the program should store logs: (logs/) ") or "logs/",
        "input_size": int(input("3. Choose an input size: (713) ") or "713"),
        "classes": int(input("4. Set the number of classes: (3) ") or "3"),
        "batch_size": int(input("5. Batch size: (2) ") or "2"),
        "weights": input("6. Stored weights: (None) ") or None
    }

    return type("", (object,), args)()


def train_network(network, args):
    """
    Training a network
    """

    print("Starting training session...\n")

    print(args.datadir)

    if network == 1:
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
    elif network == 2:
        model, eval_model, manipulate_model = CapsNet(input_shape=(args.input_size, args.input_size), n_class=3,
                                                      routings=3)
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


if __name__ == '__main__':
    # Start off by printing the menu
    print_menu()
