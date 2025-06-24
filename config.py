import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--cuda', type=bool, default=True, help='use cuda')
parser.add_argument('--seed', type=int, default=1, help='random seed')

# dataset
parser.add_argument('--root', type=str, default='./data', help='path to data set')
parser.add_argument('--batch_size', type=int, default=512, help='batch size of dataset')
parser.add_argument('--image_size', type=int, default=64, help='resize image')

# model
parser.add_argument('--model', type=str, default='ResNet', help='choose model')

# train
parser.add_argument('--load_state', type=bool, default=False, help='load state from disk')
parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate')
parser.add_argument('--result', type=str, default='./result', help='path to result')

args = parser.parse_args()
