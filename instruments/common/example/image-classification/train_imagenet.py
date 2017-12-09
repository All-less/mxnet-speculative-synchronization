import os
import argparse
import logging
role = os.getenv('DMLC_ROLE').upper()
if role == 'WORKER':
    role = 'Worker'  # backward compatibility
rank = os.getenv('DMLC_{}_ID'.format(role.upper()))
logging.basicConfig(level=logging.INFO, format='%(asctime)s {0}[{1}] %(message)s'.format(role, rank))
from common import find_mxnet, data, fit
import mxnet as mx

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train imagenet",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    # use a large aug level
    data.set_data_aug_level(parser, 3)
    parser.set_defaults(
        # network
        network          = 'resnet',
        num_layers       = 18,
        # data
        data_train       = '/home/ubuntu/ILSVRC2012/ILSVRC2012_dataset_train.rec', # ALL DATA MUST BE PLACED IN A FOLDER
        data_val         = '/home/ubuntu/ILSVRC2012/ILSVRC2012_dataset_val.rec',   # INSTEAD OF A BUCKET
        num_classes      = 1000,
        num_examples     = 281167,
        image_shape      = '3,224,224',
        min_random_scale = 1, # if input image has min size k, suggest to use
                              # 256.0/x, e.g. 0.533 for 480
        # train
        lr               = 0.03,
        num_epochs       = 80,
        lr_step_epochs   = '30,60',
        disp_batches     = 1
    )
    args = parser.parse_args()

    # load network
    from importlib import import_module
    net = import_module('symbols.'+args.network)
    sym = net.get_symbol(**vars(args))

    # train
    fit.fit(args, sym, data.get_rec_iter)
