# coding: utf-8
import os
import argparse
import logging
import threading
import random
import math

import mxnet as mx
import numpy as np
import scipy as sp
import scipy.spatial

import util


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Word Embedding with SGD",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epoch-num', type=int, default=100,
                        help='number of epochs to run')
    parser.add_argument('-b', '--batch-size', type=int, default=1000000,  # TODO
                        help='number of samples in a batch')
    parser.add_argument('-l', '--learning-rate', type=float, default=1e-6,
                        help='learning rate used in updating')
    parser.add_argument('-d', '--dimension', type=int, default=100,
                        help='dimension of embedding vectors')
    args, unknonw = parser.parse_known_args()
    return args

def prepare_data():
    data_name = os.path.join('data', 'enwiki8.npy')
    util.download_file('https://mxnet-experiment.s3.amazonaws.com/enwiki-dataset/enwiki8.npy', data_name)
    with open(data_name, 'rb') as f:
        return np.load(f)

def train(dataset, args):

    dim = args.dimension
    vocab_size = int(max(dataset[:, 0])) + 1
    size = dataset.shape[0]

    lr = args.learning_rate
    V = np.random.randn(vocab_size, dim)
    G = np.ones((vocab_size, dim), dtype='float64')
    C = random.random()
    gc = 1
    a = 0.75
    Xm = 100

    kv_V = mx.nd.array(V)
    # create kvstore
    kvstore = util.create_kvstore(kv_V)

    logging.info('Start training.')
    for epoch in range(args.epoch_num):
        epoch_start, epoch_end = util.get_epoch_interval(epoch, size)
        start, end = epoch_start, min(epoch_start + args.batch_size, epoch_end)
        loss = 0
        while True:
            d_loss = 0
            g = np.zeros((vocab_size, dim), dtype='float64')
            for i in range(start, end):
                if util.check_cancel(i, start, end):
                    logging.info('restart computation')
                    break
                # get entry
                row, col, co = dataset[i]
                row, col = int(row), int(col)
                if 1 - sp.spatial.distance.cosine(V[row], V[col]) < math.cos(math.pi * 50 / 180):
                    continue
                # compute update and loss
                t1 = C - math.log(co) + np.dot(V[row], V[col])
                t2 = t1 * (1 if co <= Xm else math.pow(co / Xm, a))
                d_loss += 0.5 * t1 * t2
                u = 2 * lr * t2 * (V[row] + V[col])
                # update local V
                u_row = u / np.sqrt(G[row])
                u_col = u / np.sqrt(G[col])
                V[row] -= u_row
                V[col] -= u_col
                # record update for updating in kvstore
                g[row] += u_row
                g[col] += u_col
                # update local G
                G[row] += np.square(u)
                G[col] += np.square(u)
                # update C
                C -= t2 / math.sqrt(gc)
                gc += t2 * t2

            # exchange with kvstore
            kv_g = mx.nd.array(g)
            util.update_param(kvstore, kv_g, kv_V, pull_only=util.need_restart())

            if not util.need_restart():
                loss += d_loss
                start, end = end, min(end + args.batch_size, epoch_end)
                if start == end:
                    break
            util.reset_cancel()

        logging.info('Epoch[{}] loss={}'.format(epoch, loss))

if __name__ == '__main__':
    util.init_logging()
    logging.info('Start executing train_embedding.py')
    args = parse_args()
    logging.info('Start preparing data.')
    dataset = prepare_data()
    train(dataset, args)
