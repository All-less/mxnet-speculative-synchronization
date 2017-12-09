import math
import mxnet as mx
import numpy as np
import mxnet.notebook.callback

import logging
logging.basicConfig(level=logging.DEBUG)

def RMSE(label, pred):
    ret = 0.0
    n = 0.0
    pred = pred.flatten()
    for i in range(len(label)):
        ret += (label[i] - pred[i]) * (label[i] - pred[i])
        n += 1.0
    return math.sqrt(ret / n)


def notebook_train(network, data_pair, num_epoch, learning_rate, optimizer='sgd', opt_args=None, ctx=[mx.gpu(0)]):
    np.random.seed(123)  # Fix random seed for consistent demos
    mx.random.seed(123)  # Fix random seed for consistent demos
    if not opt_args:
        opt_args = {}
    if optimizer=='sgd' and (not opt_args):
        opt_args['momentum'] = 0.9

    model = mx.model.FeedForward(
        ctx = ctx,
        symbol = network,
        num_epoch = num_epoch,
        optimizer = optimizer,
        learning_rate = learning_rate,
        wd = 1e-4,
        **opt_args
    )

    train, test = (data_pair)

    lc = mxnet.notebook.callback.LiveLearningCurve('RMSE', 1)
    model.fit(X = train,
              eval_data = test,
              eval_metric = RMSE,
              **mxnet.notebook.callback.args_wrapper(lc)
              )
    return lc


def dist_train(network, data_pair, batch_size, num_workers):
    # kvstore
    kv = mx.kvstore.create('dist_async')

    # data iterators
    (train, val) = data_pair

    # devices for training
    devs = mx.cpu()

    # create model
    model = mx.mod.Module(
        context     = devs,
        symbol      = network,
        data_names  = ['item', 'user'],
        label_names = ['score']
    )

    steps = [ s * num_workers * 10000000 / batch_size for s in [10, 20, 25, 30] ]
    optimizer_params = {
        'learning_rate': 0.0004,
        'momentum' : 0.9,
        'wd' : 1e-4,
        'lr_scheduler': mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=0.1)
    }

    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)

    # evaluation metrices
    eval_metrics = ['rmse']

    # callbacks that run after each batch
    batch_end_callbacks = [mx.callback.Speedometer(batch_size, 100000 / batch_size)]

    # run
    model.fit(train,
        begin_epoch        = 0,
        num_epoch          = 200,
        eval_data          = val,
        eval_metric        = eval_metrics,
        kvstore            = kv,
        optimizer          = 'sgd',
        optimizer_params   = optimizer_params,
        initializer        = initializer,
        batch_end_callback = batch_end_callbacks,
        allow_missing      = True
    )
