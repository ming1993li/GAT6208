import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--gpu-ids', help='GPU IDs', type=str, default='0')
parser.add_argument('--seed', type=int, default=666, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--val-interval', type=int, default=20, help='Interval epochs for validation.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--num_gals_in', type=int, default=8, help='Number of graph attentional layers for input encoding.')
parser.add_argument('--num_gals_predict', type=int, default=1, help='Number of graph attentional layers for class prediction.')
parser.add_argument('--c_dim', type=int, default=8, help='hidden dimention of graph attentional layers.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

args = parser.parse_args()