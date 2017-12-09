# coding: utf-8
import subprocess
import os
import threading
import errno
import logging

import mxnet as mx


NUM_WORKER = os.getenv('DMLC_NUM_WORKER')
WORKER_ID = os.getenv('DMLC_WORKER_ID')

# a global variable for cross-threading synchronization
_need_cancellation = threading.Event()
CANCEL_VARS = {
    'is_restart': False,
    'need_restart': False,
    'enable_cancel': int(os.getenv('MXNET_ENABLE_CANCEL') or '0')
}

def cancel_callback():
    if not _need_cancellation.is_set():
        _need_cancellation.set()

def reset_cancellation():
    _need_cancellation.clear()
    return time.time()

def check_cancel(index, start, end):
    if CANCEL_VARS['enable_cancel'] \
        and not CANCEL_VARS['is_restart'] \
        and end - start > 1 \
        and (index - start) / (end - start) < 2 \
        and (end - start) / 100 > 1 \
        and (index - start) % ((end - start) / 100) == 0 \
        and _need_cancellation.is_set():
        CANCEL_VARS['need_restart'] = True
        return True
    else:
        return False

def need_restart():
    return CANCEL_VARS['need_restart']

def init_logging():
    if NUM_WORKER is None:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s Worker[{}] %(message)s'.format(WORKER_ID)
        )

def reset_cancel():
    _need_cancellation.clear()
    CANCEL_VARS['is_restart'] = CANCEL_VARS['need_restart']
    CANCEL_VARS['need_restart'] = False

def init_cancel(kvstore):
    if CANCEL_VARS['enable_cancel']:
        kvstore.set_cancel_callback(cancel_callback)

def create_kvstore(initial_value):
    kvstore = mx.kvstore.create('dist_async')
    kvstore.init(0, initial_value)
    kvstore.set_optimizer(mx.optimizer.SGD(learning_rate=-1))
    init_cancel(kvstore)
    return kvstore

def update_param(kvstore, gradient, param, pull_only=False):
    if not pull_only:
        logging.info('PUSH')
        kvstore.push(0, gradient, last=True)
        logging.info('PUSHED')
    logging.info('PULL')
    kvstore.pull(0, param)
    logging.info('PULLED')

def split_data(dataset):
    if NUM_WORKER is None:  # not distributed training
        return dataset, (0, dataset.shape[0])
    else:
        l = dataset.shape[0]
        num, index = int(NUM_WORKER), int(WORKER_ID)
        start = l / num * index
        end = min(l / num * (index + 1), l)
        return dataset[start:end], (start, end)

def download_file(url, local_fname=None, force_write=False):
    # requests is not default installed
    import requests
    if local_fname is None:
        local_fname = url.split('/')[-1]
    if not force_write and os.path.exists(local_fname):
        return local_fname

    dir_name = os.path.dirname(local_fname)

    if dir_name != "":
        if not os.path.exists(dir_name):
            try: # try to create the directory if it doesn't exists
                os.makedirs(dir_name)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

    r = requests.get(url, stream=True)
    assert r.status_code == 200, "failed to open %s" % url
    with open(local_fname, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
    return local_fname

def get_gpus():
    """
    return a list of GPUs
    """
    try:
        re = subprocess.check_output(["nvidia-smi", "-L"], universal_newlines=True)
    except OSError:
        return []
    return range(len([i for i in re.split('\n') if 'GPU' in i]))

def get_epoch_interval(epoch_num, data_size):
    if WORKER_ID is None:
        return 0, data_size
    else:
        num, index = int(NUM_WORKER), int(WORKER_ID)
        shard_size = data_size / num
        epoch_start = (shard_size * (index + epoch_num)) % data_size
        epoch_end = min(epoch_start + shard_size, data_size)
        return epoch_start, epoch_end
