# coding: utf-8
import logging
import os
import pathlib
import argparse
import subprocess
import shutil


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
    parser.add_argument('-o', '--extra-dir', type=str,
                        help='directory containing extra instrumentation source')
    args, unknown = parser.parse_known_args()
    return args

def is_c_source(full_path):
    for suffix in [ '.h', '.cc', '.cpp', '.c', '.hpp' ]:
        if full_path.endswith(suffix):
            return True
    return False

def need_recompile(full_path, need_recompile, need_recompile_dmlc_core):
    if need_recompile_dmlc_core or (is_c_source(full_path) and 'dmlc-core' in full_path):
        return True, True
    elif need_recompile or is_c_source(full_path):
        return True, False
    else:
        return False, False

def recompile_mxnet():
    logging.info('Start recompiling mxnet.')
    if subprocess.run(f'cd {get_target_dir()} && make -s -j4', shell=True):
        logging.info('Successfully recompiled mxnet.')
    else:
        logging.warn('Recompiling mxnet failed.')

def recompile_dmlc_core():
    logging.info('Start recompiling dmlc-core.')
    if subprocess.run(f'cd {get_target_dir()}/dmlc-core && make -s -j4', shell=True):
        logging.info('Successfully recompiled dmlc-core.')
    else:
        logging.warn('Recompiling dmlc-core failed.')

def copy_files(instrument_dir, cpl, cpl_core):
    for dir_path, dirs, files in os.walk(instrument_dir):
        for file in files:
            full_path = f'{dir_path}/{file}'
            try:
                target_dir = dir_path.replace(instrument_dir, str(MXNET_DIR))
                _mkdir(target_dir)  # ensure target dir exists
                shutil.copyfile(full_path, f'{target_dir}/{file}')
                logging.info(f'Copied {dir_path.replace(instrument_dir, "")}/{file}.')
                cpl, cpl_core = need_recompile(full_path, cpl, cpl_core)
            except Exception as e:
                logging.warning(f'Error copying {dir_path.replace(instrument_dir, "")}/{file}.')
                logging.warning('{}'.format(e))
    return cpl, cpl_core

def instrument_source():
    args = get_args()
    cpl, cpl_core = False, False  # need_recompile, need_recompile_dmlc_core
    logging.info('Start walking in instrumentation directory.')
    cpl, cpl_core = copy_files(f'{INSTRUMENT_DIR}/{args.base_dir}', cpl, cpl_core)
    if args.extra_dir:
        cpl, cpl_core = copy_files(f'{INSTRUMENT_DIR}/{args.extra_dir}', cpl, cpl_core)
    if cpl_core:
        recompile_dmlc_core()
    if cpl:
        recompile_mxnet()

if __name__ == '__main__':
    instrument_source()
