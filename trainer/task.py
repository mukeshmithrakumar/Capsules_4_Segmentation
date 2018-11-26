import argparse
import sys
import os

if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    __package__ = "GCaps.trainer"

    from ..trainer import train  # Your model.py file.

    """ 
    Parse the arguments.
    """
    # python task.py dynamic_routing

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network.")
    parser.add_argument('--routing', type=str.lower, default='em_routing')
    # parser.add_argument('routing', choices=['dynamic_routing', 'em_routing'], type=str.lower)

    parser.add_argument('--batch_size',
                        default=100,
                        type=int,
                        help='batch size'
                        )

    parser.add_argument('--epochs',
                        default=2,
                        type=int
                        )

    parser.add_argument('--lr',
                        default=0.001,
                        type=float,
                        help="Initial learning rate"
                        )

    parser.add_argument('--lr_decay',
                        default=0.9,
                        type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs"
                        )

    parser.add_argument('--lam_recon',
                        default=0.392,
                        type=float,
                        help="The coefficient for the loss of decoder"
                        )

    parser.add_argument('-r', '--iterations',
                        default=3,
                        type=int,
                        help="Number of iterations used in routing algorithm. should > 0"
                        )

    parser.add_argument('--shift_fraction',
                        default=0.1,
                        type=float,
                        help="Fraction of pixels to shift at most in each direction."
                        )

    parser.add_argument('--debug',
                        action='store_true',
                        help="Save weights by TensorBoard"
                        )

    parser.add_argument('--save_dir',
                        default='./result'
                        )

    parser.add_argument('-t', '--testing',
                        action='store_true',
                        help="Test the trained model on testing dataset"
                        )

    parser.add_argument('--digit',
                        default=5,
                        type=int,
                        help="Digit to manipulate"
                        )

    parser.add_argument('-w', '--weights',
                        default=None,
                        help="The path of the saved weights. Should be specified when testing"
                        )

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train.train(args)
