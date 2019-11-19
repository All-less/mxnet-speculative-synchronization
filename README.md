# mxnet-speculative-synchronization

A new parallel scheme implemented on MXNet.

## Prerequisites

- Python 3.5+ (for instrumenting script)
- Python 2.7+ (for starting MXNet)
- make
- gcc

## Installation

Run following commands to get the source code.

```bash
git clone --recursive https://github.com/All-less/mxnet-speculative-synchronization.git
cd mxnet-speculative-synchronization
```

Roll back MXNet to commit `7fcaf15a`.

```bash
cd mxnet
git checkout 7fcaf15a3a597cc72a342d1bdb00273dec00e78c
git submodule update --recursive
```

Our implementation is based on MXNet, so we need to insert some instrumentation into MXNet sources. We will elaborate on `extra-dir` option in [next section](#Run).

```bash
python instrument_source.py --extra-dir <fixed_waiting|freshness_tuning>
```

After instrumenting, follow the instructions [here](scripts/install-mxnet.sh) to build MXNet.

## Get Started

The training process is the same as [original](http://34.201.8.176/versions/0.11.0/tutorials/vision/large_scale_classification.html), whereas you need to set some environment variables to activate speculative synchronization. We provide two different modes of synchronization.

### Fixed Waiting

In *Fixed Waiting* mode, you need to specify how long each worker will wait and how many fresh updates to trigger synchronization.

```bash
export MXNET_ENABLE_CANCEL=1     # enable speculative synchronization
export MXNET_WAIT_RATIO=0.10     # wait 10% of batch time
export MXNET_CANCEL_THRESHOLD=5  # synchronize when getting more than 5 fresh updates.
```

### Freshness Tuning

In *Freshness Tuning* mode, you only need to turn on the switch.

```bash
export MXNET_ENABLE_CANCEL=1
```

## Caveats

1. Only CPU training is supported.
2. In case of any conflict caused by upstream updating, please use MXNet with commit `11fe466`.
