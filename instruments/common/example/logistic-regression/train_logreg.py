# coding: utf-8
import os
import argparse
import logging
import math

import mxnet as mx
import numpy as np

import util


role = os.getenv('DMLC_ROLE').upper()
rank = os.getenv('DMLC_{}_ID'.format(role))
logging.basicConfig(level=logging.INFO, format='%(asctime)s {0}[{1}] %(message)s'.format(role, rank))


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Logistic Regression on the given data.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epoch-num', type=int, default=100,
                        help='number of epochs to run')
    parser.add_argument('-b', '--batch-size', type=int, default=5000000,  # TODO
                        help='number of samples in a batch')
    parser.add_argument('-l', '--learning-rate', type=float, default=1e-10,
                        help='learning rate used in updating')
    parser.add_argument('-n', '--num-workers', type=int, default=30,
                        help='number of workers')
    args, unknonw = parser.parse_known_args()
    return args

def prepare_data(args):
    global rank
    if rank is None:
        rank = '0'
    train_data_name = os.path.join('data', 'train_data_{}.csv'.format(rank))
    util.download_file('https://mxnet-experiment.s3.amazonaws.com/criteo-dataset/train_data_{}.csv'.format(rank), train_data_name)
    train_label_name = os.path.join('data', 'train_label_{}.csv'.format(rank))
    util.download_file('https://mxnet-experiment.s3.amazonaws.com/criteo-dataset/train_label_{}.csv'.format(rank), train_label_name)
    eval_data_name = os.path.join('data', 'eval_data_{}.csv'.format(rank))
    # util.download_file('https://mxnet-experiment.s3.amazonaws.com/criteo-dataset/eval_data_{}.csv'.format(rank), eval_data_name)
    eval_label_name = os.path.join('data', 'eval_label_{}.csv'.format(rank))
    # util.download_file('https://mxnet-experiment.s3.amazonaws.com/criteo-dataset/eval_label_{}.csv'.format(rank), eval_label_name)
    os.system('head -n 5000000 {} > {}'.format(train_data_name, eval_data_name))
    os.system('head -n 5000000 {} > {}'.format(train_label_name, eval_label_name))
    train_data = mx.io.CSVIter(data_name='data', data_csv=train_data_name, data_shape=(12,),
                               label_name='target', label_csv=train_label_name, label_shape=(1,),
                               batch_size=args.batch_size)
    eval_data = mx.io.CSVIter(data_name='data', data_csv=eval_data_name, data_shape=(12,),
                              label_name='target', label_csv=eval_label_name, label_shape=(1,),
                              batch_size=args.batch_size)
    return train_data, eval_data

def get_symbol():
    data = mx.sym.Variable('data')
    target = mx.sym.Variable('target')
    fc = mx.sym.FullyConnected(data=data, num_hidden=1, name='fc')
    pred = mx.sym.LogisticRegressionOutput(data=fc, label=target)
    return pred

def train(symbol, train_data, eval_data, args):
    # kvstore
    kv = mx.kvstore.create('dist_async')

    # devices for training
    devs = mx.cpu()

    # create model
    model = mx.mod.Module(
        context     = devs,
        symbol      = symbol,
        data_names  = [ 'data' ],
        label_names = [ 'target' ]
    )

    steps = [ s * args.num_workers * 90000000 / args.batch_size for s in [10, 20, 25, 30] ]
    optimizer_params = {
        'learning_rate': 0.000005,
        'momentum' : 0.9,
        'wd' : 1e-4,
        'lr_scheduler': mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=0.3)
    }

    # callbacks that run after each batch
    # batch_end_callbacks = [mx.callback.Speedometer(batch_size, 100000 / batch_size)]

    # run
    model.fit(train_data,
        begin_epoch        = 0,
        num_epoch          = args.epoch_num,
        eval_data          = eval_data,
        eval_metric        = [ 'rmse' ],
        kvstore            = kv,
        optimizer          = 'sgd',
        optimizer_params   = optimizer_params,
        initializer        = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        # batch_end_callback = batch_end_callbacks,
        allow_missing      = True
    )

def train_logreg():
    logging.info('Start executing train_logreg.py')
    args = parse_args()
    symbol = get_symbol()
    logging.info('Start preparing data')
    train_data, eval_data = prepare_data(args)
    logging.info('Start training')
    train(symbol, train_data, eval_data, args)

if __name__ == '__main__':
    train_logreg()
