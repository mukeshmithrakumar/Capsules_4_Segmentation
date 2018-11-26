import numpy as np
from preprocessing.load import load_data

from dynamic_routing.capsnet_dr import DRCapsules, create_model, manipulate_latent, test
from em_routing.capsnet_em import EMCapsules, create_arch


def train(args):
    # load data
    (x_train, y_train), (x_test, y_test) = load_data()

    # define model
    # choose between dynamic routing and em routing
    if args.routing == 'dynamic_routing':
        model, eval_model, manipulate_model = DRCapsules(input_shape=x_train.shape[1:],
                                                         n_class=len(np.unique(np.argmax(y_train, 1))),
                                                         iterations=args.iterations)
        model.summary()

        # train or test
        if args.weights is not None:  # init the model weights with provided one
            model.load_weights(args.weights)
        if not args.testing:
            create_model(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
        else:  # as long as weights are given, will run testing
            if args.weights is None:
                print('No weights are provided. Will test using random initialized weights.')
            manipulate_latent(manipulate_model, (x_test, y_test), args)
            test(model=eval_model, data=(x_test, y_test), args=args)

    elif args.routing == 'em_routing':
        model, eval_model, manipulate_model = EMCapsules(input_shape=x_train.shape[1:],
                                                         n_class=len(np.unique(np.argmax(y_train, 1))),
                                                         iterations=args.iterations)
        model.summary()

        # train or test
        if args.weights is not None:  # init the model weights with provided one
            model.load_weights(args.weights)
        if not args.testing:
            create_arch(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
        else:  # as long as weights are given, will run testing
            if args.weights is None:
                print('No weights are provided. Will test using random initialized weights.')
            manipulate_latent(manipulate_model, (x_test, y_test), args)
            test(model=eval_model, data=(x_test, y_test), args=args)

    else:
        raise ValueError('You have specified a incorrect routing protocol')


print("Done")
