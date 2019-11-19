# This script installs MXNet without GPU support on Ubuntu Server 16.04 LTS.

set -e

# downgrade to gcc-4.8 as gcc-5.4 will cause 'internal compiler error'
sudo apt-get -yq update
sudo apt-get -yq install gcc-4.8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 10

# install dependencies
sudo apt-get -yq update
sudo apt-get -yq install python-dev python-numpy python-pip libatlas-base-dev build-essential git
sudo apt-get -yq install libcurl4-openssl-dev libssl-dev  # required by S3

# install optional dependency: OpenCV
sudo apt-get -yq install libopencv-dev

# build shared library
cd mxnet/dmlc-core
cp make/config.mk make/config.mk.bak
cat make/config.mk.bak \
    | sed 's/USE_S3 = 0/USE_S3 = 1/g' > make/config.mk
make -s -j4
cd ..

# edit make/config.mk, enable USE_DIST_KVSTORE, USE_S3
cp make/config.mk make/config.mk.bak
cat make/config.mk.bak \
    | sed 's/USE_PROFILER =/USER_PROFILER = 1/g' \
    | sed 's/USE_DIST_KVSTORE = 0/USE_DIST_KVSTORE = 1/g' \
    | sed 's/USE_S3 = 0/USE_S3 = 1/g' > make/config.mk

make -s -j4

# install python package
cd python
sudo python setup.py install

# required by examples in mxnet/example/image-classification
sudo pip install -q requests
