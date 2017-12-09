# coding: utf-8
import time
import os
import logging
import sys

import mxnet as mx


kv = mx.kvstore.create('dist_async')
kv.set_optimizer(mx.optimizer.SGD(learning_rate=-1))
shape = (2, 3)
kv.init(0, mx.nd.ones(shape))
role = os.getenv('DMLC_ROLE').upper()
rank = os.getenv('DMLC_{}_ID'.format(role))


while True:
    kv.push(0, mx.nd.ones(shape))
    out = mx.nd.zeros(shape)
    time.sleep(3)
    kv.pull(0, out=out)
