# coding: utf-8
import logging
import os
import pathlib
import argparse
import subprocess
import shutil


logging.basicConfig(level=logging.INFO)
ROOT_DIR = pathlib.Path(__file__).parent
INSTRUMENT_DIR = ROOT_DIR / 'instruments'
MXNET_DIR = ROOT_DIR / 'mxnet'


def _mkdir(newdir):
    """
    works the way a good mkdir should :)
        - already exists, silently complete
        - regular file in the way, raise an exception
        - parent directory(ies) does not exist, make them as well
    """
    if type(newdir) is not str:
        newdir = str(newdir)
    if os.path.isdir(newdir):
        pass
    elif os.path.isfile(newdir):
        raise OSError("a file with the same name as the desired " \
                      "dir, '%s', already exists." % newdir)
    else:
        head, tail = os.path.split(newdir)
        if head and not os.path.isdir(head):
            _mkdir(head)
        if tail:
            os.mkdir(newdir)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base-dir', type=str, default='common',
                        help='directory containing base instrumentation source')
    parser.add_argument('-e', '--extra-dir', type=str,
                        help='directory containing extra instrumentation source')
    args, unknown = parser.parse_known_args()
    return args

def copy_files(instrument_dir):
    for dir_path, dirs, files in os.walk(instrument_dir):
        for file in files:
            full_path = f'{dir_path}/{file}'
            try:
                target_dir = dir_path.replace(instrument_dir, str(MXNET_DIR))
                _mkdir(target_dir)  # ensure target dir exists
                shutil.copyfile(full_path, f'{target_dir}/{file}')
                logging.info(f'Copied {dir_path.replace(instrument_dir, "")}/{file}.')
            except Exception as e:
                logging.warning(f'Error copying {dir_path.replace(instrument_dir, "")}/{file}.')
                logging.warning('{}'.format(e))

def instrument_source():
    args = get_args()
    logging.info('Start walking in instrumentation directory.')
    copy_files(f'{INSTRUMENT_DIR}/{args.base_dir}')
    if args.extra_dir:
        copy_files(f'{INSTRUMENT_DIR}/{args.extra_dir}')

if __name__ == '__main__':
    instrument_source()
