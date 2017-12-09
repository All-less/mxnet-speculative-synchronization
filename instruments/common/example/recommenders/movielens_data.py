"""MovieLens data handling: download, parse, and expose as DataIter
"""
import logging
import os
import mxnet as mx


def load_mldata_iter(filename, batch_size, delimiter='\t'):
    """Not particularly fast code to parse the text file and load it into three NDArray's
    and product an NDArrayIter
    """
    user = []
    item = []
    score = []
    with file(filename) as f:
        for line in f:
            tks = line.strip().split(delimiter)
            if len(tks) != 4:
                continue
            user.append(int(tks[0]))
            item.append(int(tks[1]))
            score.append(float(tks[2]))
    user = mx.nd.array(user)
    item = mx.nd.array(item)
    score = mx.nd.array(score)
    ret = mx.io.NDArrayIter(data={'user':user,'item':item},label={'score':score},
                             batch_size=batch_size, shuffle=True)
    return ret

def ensure_local_data(prefix):
    if not os.path.exists("%s.zip" % prefix):
        print("Downloading MovieLens data: %s" % prefix)
        os.system("wget http://files.grouplens.org/datasets/movielens/%s.zip" % prefix)
        os.system("unzip %s.zip" % prefix)


def get_data_iter(batch_size, prefix='ml-100k'):
    """Returns a pair of NDArrayDataIter, one for train, one for test.
    """
    ensure_local_data(prefix)
    return (load_mldata_iter('./%s/u1.base' % prefix, batch_size),
            load_mldata_iter('./%s/u1.test' % prefix, batch_size))

def max_id(fname, delimiter='\t'):
    mu = 0
    mi = 0
    for line in file(fname):
        tks = line.strip().split(delimiter)
        if len(tks) != 4:
            continue
        mu = max(mu, int(tks[0]))
        mi = max(mi, int(tks[1]))
    return mu + 1, mi + 1

def get_ml100k_data_iter(batch_size):
    return get_data_iter(batch_size)

def ensure_ml10m_data():
    if not os.path.exists('ml-10m.zip'):
        logging.info('Downloading MovieLens 10M dataset ...')
        os.system('wget -nv http://files.grouplens.org/datasets/movielens/ml-10m.zip')
    if not os.path.isdir('ml-10M100K'):
        logging.info('Unpacking dataset ...')
        os.system('sudo apt install -yq unzip && unzip ml-10m.zip')
    if not os.path.exists('ml-10M100K/r1.train'):
        logging.info('Splitting dataset ...')
        os.system('cd ml-10M100K && ./split_ratings.sh')

def get_ml10m_data_iter(batch_size):
    ensure_ml10m_data()
    return (
        load_mldata_iter('ml-10M100K/r1.train', batch_size, delimiter='::'),
        load_mldata_iter('ml-10M100K/r1.test', batch_size, delimiter='::')
    )
