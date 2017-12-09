# coding: utf-8
import os
import argparse
import logging
role = os.getenv('DMLC_ROLE').upper()
rank = os.getenv('DMLC_{}_ID'.format(role))
logging.basicConfig(level=logging.INFO, format='%(asctime)s {0}[{1}] %(message)s'.format(role, rank))

import mxnet as mx
from movielens_data import get_ml10m_data_iter, max_id
from matrix_fact import dist_train


def get_mf_network(user_dim, item_dim, output_dim, hidden):
    # input
    user = mx.symbol.Variable('user')
    item = mx.symbol.Variable('item')
    score = mx.symbol.Variable('score')
    # user latent features
    user = mx.symbol.Embedding(data=user, input_dim=user_dim, output_dim=output_dim)
    user = mx.symbol.Activation(data=user, act_type='relu')
    user = mx.symbol.FullyConnected(data=user, num_hidden=hidden)
    # item latent features
    item = mx.symbol.Embedding(data=item, input_dim=item_dim, output_dim=output_dim)
    item = mx.symbol.Activation(data=item, act_type='relu')
    item = mx.symbol.FullyConnected(data=item, num_hidden=hidden)
    # predict by the inner product
    pred = user * item
    pred = mx.symbol.sum_axis(data=pred, axis=1)
    pred = mx.symbol.Flatten(data=pred)
    # loss layer
    pred = mx.symbol.LinearRegressionOutput(data=pred, label=score)
    return pred

def train_mf():
    parser = argparse.ArgumentParser(description="train mf")
    parser.add_argument('--batch-num', type=int, default=100000,
                        help='number of samples in a batch')
    parser.add_argument('--num-workers', type=int, default=24,
                        help='number or workers')
    args, unknonw = parser.parse_known_args()
    data_pair = get_ml10m_data_iter(batch_size=args.batch_num)
    max_user, max_item = max_id('./ml-10M100K/ratings.dat', delimiter='::')
    network = get_mf_network(max_user, max_item, 64, 64)
    dist_train(network, data_pair, args.batch_num, args.num_workers)

if __name__ == '__main__':
    train_mf()
