import argparse
import math


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser(
        description='Pytorch Detecting Out-of-distribution examples in neural networks')

    parser.add_argument('--in-dataset', default="CIFAR-10",
                        type=str, help='CIFAR-10 imagenet')
    parser.add_argument('--out-datasets', default=['inat', 'sun50', 'places50', 'dtd', ], nargs="*", type=str,
                        help="['SVHN', 'LSUN', 'iSUN', 'dtd', 'places365']  ['inat', 'sun50', 'places50', 'dtd', ]")
    parser.add_argument('--backbone', default='resnet34',
                        type=str, help='model backbone')
    parser.add_argument('--method', default='supcon',
                        type=str, help='method used for training')
    parser.add_argument('--seed', default=1, type=int, help='seed')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device')

    # Optimization options
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=128,
                        type=int, help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=0.0001, type=float,
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--print_every', default=50, type=int,
                        help='print model status')
    parser.add_argument('--fine_tune', action='store_true', default=False,
                        help='fine_tuning')
    parser.add_argument('--temp', default=0.1, type=float,
                        help='temperature')
    parser.add_argument('--cosine', action='store_true', default=True,
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')

    # backbone options
    parser.add_argument('--layers', default=100, type=int,
                        help='total number of layers (default: 100)')
    parser.add_argument('--depth', default=40, type=int,
                        help='depth of resnet')
    parser.add_argument('--width', default=4, type=int, help='width of resnet')
    parser.add_argument('--growth', default=12, type=int,
                        help='number of new channels per layer (default: 12)')
    parser.add_argument('--droprate', default=0.0, type=float,
                        help='dropout probability (default: 0.0)')
    parser.add_argument('--save-path', default="CIFAR-10-ResNet18.pt",
                        type=str, help="the path to save the trained model")
    parser.add_argument('--reduce', default=0.5, type=float,
                        help='compression rate in transition stage (default: 0.5)')
    parser.add_argument('--score', default="mahalanobis", type=str,
                        help='the scoring function for evaluation')
    parser.add_argument('--threshold', default=1.0,
                        type=float, help='sparsity level')
    parser.set_defaults(argument=True)
    
    # prototypes arguments
    parser.add_argument('--k', default=5, type=int)
    parser.add_argument('--momentum', default=0.9, type=float, help="SGD momentum")
    parser.add_argument('--proto_m', default=0.999, type=float, help="prototypes update momentum")
    parser.add_argument('--cache-size', default=6, type=int)
    parser.add_argument('--nviews', default=2, type=int)
    
    # loss config
    parser.add_argument('--lambda_pcon', default=1., type=float)
    parser.add_argument('--epsilon', default=0.05, type=float)
    
    args = parser.parse_args()
    
    if "noaug" in args.method:
        args.nviews = 1

    if args.batch_size > 256:
        args.warm = True
    if args.warm:
        args.warmup_from = 0.01
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.lr * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.lr - eta_min) * (
                1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.lr

    return args

