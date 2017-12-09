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
import scipy.sparse

import util


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Graph Eigenvector with SVRG",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epoch-num', type=int, default=1000,
                        help='number of epochs to run')
    parser.add_argument('-b', '--batch-size', type=int, default=100000,  # TODO
                        help='number of samples in a batch')
    parser.add_argument('-l', '--learning-rate', type=float, default=1e-4,
                        help='learning rate used in updating')
    args, unknonw = parser.parse_known_args()
    return args

def prepare_data():
    data_name = os.path.join('data', 'friendster-300K.npz')
    t_name = os.path.join('data', 'friendster-300K-T.npz')
    x_name = os.path.join('data', 'friendster-300K-x.npy')
    b_name = os.path.join('data', 'friendster-300K-b.npy')
    util.download_file('https://mxnet-experiment.s3.amazonaws.com/friendster-dataset/friendster-300K.npz', data_name)
    util.download_file('https://mxnet-experiment.s3.amazonaws.com/friendster-dataset/friendster-300K-T.npz', t_name)
    util.download_file('https://mxnet-experiment.s3.amazonaws.com/friendster-dataset/friendster-300K-x.npy', x_name)
    util.download_file('https://mxnet-experiment.s3.amazonaws.com/friendster-dataset/friendster-300K-b.npy', b_name)
    return sp.sparse.load_npz(data_name), sp.sparse.load_npz(t_name), np.load(x_name), np.load(b_name)

def train(data, args):

    dataset, T, x, b = data
    dim = dataset.shape[0]
    shard, interval = util.split_data(dataset)
    size = shard.shape[0]
    kv_x = mx.nd.zeros((dim, 1))
    kv_d = mx.nd.zeros((dim, 1))

    # create kvstore
    kvstore = util.create_kvstore(kv_x)

    A = dataset

    lambda_ = np.dot((x.T * A.T), (A * x))[0][0]
    gamma = args.learning_rate

    logging.info('Start training.')
    for epoch in range(args.epoch_num):
        # gradient in this epoch
        t1 = lambda_ / dim * x - b / dim
        nnz = A.getnnz(0).reshape((dim, 1))
        g = (np.multiply(nnz, t1) + t1 * dim + T * x) / dim

        start, end = 0, min(args.batch_size, size)
        while True:
            x_prime = x.copy()
            for i in range(start, end):
                if util.check_cancel(i, start, end):
                    logging.info('restart computation')
                    break
                # compute update
                u = lambda_ / dim * gamma * x - gamma * (g - lambda_ / dim * x) - gamma * np.sum(shard[i] * (x - x_prime)) * shard[i].T
                # update local vector
                x = x - u
                kv_d = kv_d - mx.nd.array(u.getA())

            # exchange with kvstore
            util.update_param(kvstore, kv_d, kv_x, pull_only=util.need_restart())
            kv_d = mx.nd.zeros((dim, 1))
            x = kv_x.asnumpy()

            if not util.need_restart():
                start, end = end, min(end + args.batch_size, size)
                if start == end:
                    break
            util.reset_cancel()

        # compute objective
        loss = size / dim * np.dot((lambda_ / 2 * x - b).T, x)
        for i in range(*interval):
            loss -= (A[i] * x) ** 2 / 2
        logging.info('Epoch[{}] loss={}'.format(epoch, np.sum(loss) + 2))

if __name__ == '__main__':
    util.init_logging()
    logging.info('Start executing train_eigen.py')
    args = parse_args()
    logging.info('Start preparing data.')
    data = prepare_data()
    train(data, args)
