# coding: utf-8
""" Key value store interface of MXNet for parameter synchronization."""
from __future__ import absolute_import

import ctypes
import pickle
from .ndarray import NDArray
from .base import _LIB
from .base import check_call, c_array, c_str, string_types, mx_uint, py_str
from .base import NDArrayHandle, KVStoreHandle
from . import optimizer as opt

def _ctype_key_value(keys, vals):
    """
    Returns ctype arrays for the key-value args. For internal use.
    """
    if isinstance(keys, int):
        if isinstance(vals, NDArray):
            return (c_array(ctypes.c_int, [keys]),
                    c_array(NDArrayHandle, [vals.handle]))
        else:
            for value in vals:
                assert(isinstance(value, NDArray))
            return (c_array(ctypes.c_int, [keys] * len(vals)),
                    c_array(NDArrayHandle, [value.handle for value in vals]))
    else:
        assert(len(keys) == len(vals))
        for k in keys:
            assert(isinstance(k, int))
        c_keys = []
        c_vals = []
        for key, val in zip(keys, vals):
            c_key_i, c_val_i = _ctype_key_value(key, val)
            c_keys += c_key_i
            c_vals += c_val_i
        return (c_array(ctypes.c_int, c_keys), c_array(NDArrayHandle, c_vals))


def _updater_wrapper(updater):
    """A wrapper for the user-defined handle."""
    def updater_handle(key, lhs_handle, rhs_handle, _):
        """ ctypes function """
        lhs = NDArray(NDArrayHandle(lhs_handle))
        rhs = NDArray(NDArrayHandle(rhs_handle))
        updater(key, lhs, rhs)
    return updater_handle


class KVStore(object):
    """A key-value store for synchronization of values, over multiple devices."""
    def __init__(self, handle):
        """Initializes a new KVStore.

        Parameters
        ----------
        handle : KVStoreHandle
            `KVStore` handle of C API.
        """
        assert isinstance(handle, KVStoreHandle)
        self.handle = handle
        self._updater = None
        self._updater_func = None

    def __del__(self):
        check_call(_LIB.MXKVStoreFree(self.handle))

    def init(self, key, value):
        """ Initializes a single or a sequence of key-value pairs into the store.

        For each key, one must `init` it before calling `push` or `pull`.
        When multiple workers invoke `init` for the same key, only
        the value supplied by worker with rank `0` is used. This function returns
        after data has been initialized successfully.

        Parameters
        ----------
        key : int or sequence of int
            The keys.
        value : NDArray or sequence of NDArray
            Values corresponding to the Keys

        Examples
        --------
        >>> # init a single key-value pair
        >>> shape = (2,3)
        >>> kv = mx.kv.create('local')
        >>> kv.init(3, mx.nd.ones(shape)*2)
        >>> a = mx.nd.zeros(shape)
        >>> kv.pull(3, out=a)
        >>> print a.asnumpy()
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]

        >>> # init a list of key-value pairs
        >>> keys = [5, 7, 9]
        >>> kv.init(keys, [mx.nd.ones(shape)]*len(keys))
        """
        ckeys, cvals = _ctype_key_value(key, value)
        check_call(_LIB.MXKVStoreInit(
            self.handle, mx_uint(len(ckeys)), ckeys, cvals))

    def push(self, key, value, priority=0, last=0):
        """ Pushes a single or a sequence of key-value pairs into the store.

        This function returns immediately after adding an operator to the engine.
        The actual operation is executed asynchronously after all previous `push`
        and `pull` calls for the same input key(s) are finished.
        There is no synchronization between workers. One can use ``_barrier()``
        to sync all workers.

        Parameters
        ----------
        key : int or list of int
            Keys

        value : NDArray or list of NDArray or list of list of NDArray
            Values corresponding to the Keys

        priority : int, optional
            The priority of the push operation.
            Higher priority push operations are likely to be executed before
            other push actions

        Examples
        --------
        >>> # push a single key-value pair
        >>> kv.push(3, mx.nd.ones(shape)*8)
        >>> kv.pull(3, out=a) # pull out the value
        >>> print a.asnumpy()
        [[ 8.  8.  8.]
        [ 8.  8.  8.]]

        >>> # aggregate the value and the push
        >>> gpus = [mx.gpu(i) for i in range(4)]
        >>> b = [mx.nd.ones(shape, gpu) for gpu in gpus]
        >>> kv.push(3, b)
        >>> kv.pull(3, out=a)
        >>> print a.asnumpy()
        [[ 4.  4.  4.]
        [ 4.  4.  4.]]

        >>> # push a list of keys.
        >>> # single device
        >>> kv.push(keys, [mx.nd.ones(shape)]*len(keys))
        >>> b = [mx.nd.zeros(shape)]*len(keys)
        >>> kv.pull(keys, out=b)
        >>> print b[1].asnumpy()
        [[ 1.  1.  1.]
        [ 1.  1.  1.]]

        >>> # multiple devices:
        >>> b = [[mx.nd.ones(shape, gpu) for gpu in gpus]] * len(keys)
        >>> kv.push(keys, b)
        >>> kv.pull(keys, out=b)
        >>> print b[1][1].asnumpy()
        [[ 4.  4.  4.]
        [ 4.  4.  4.]]
        """
        ckeys, cvals = _ctype_key_value(key, value)
        check_call(_LIB.MXKVStorePush(
            self.handle, mx_uint(len(ckeys)), ckeys, cvals,
            ctypes.c_int(priority), ctypes.c_int(last)))

    def pull(self, key, out=None, priority=0):
        """ Pulls a single value or a sequence of values from the store.

        This function returns immediately after adding an operator to the engine.
        Subsequent attempts to read from the `out` variable will be blocked until the
        pull operation completes.

        `pull` is executed asynchronously after all previous `push` and `pull` calls
        for the same input key(s) are finished.

        The returned values are gauranteed to the latest values in the store.

        Parameters
        ----------
        key : int or list of int
            Keys.

        out: NDArray or list of NDArray or list of list of NDArray
            Values corresponding to the Keys.

        priority : int, optional
            The priority of the pull operation.
            Higher priority pull operations are likely to be executed before
            other pull actions

        Examples
        --------
        >>> # pull a single key-value pair
        >>> a = mx.nd.zeros(shape)
        >>> kv.pull(3, out=a)
        >>> print a.asnumpy()
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]

        >>> # pull into multiple devices
        >>> b = [mx.nd.ones(shape, gpu) for gpu in gpus]
        >>> kv.pull(3, out=b)
        >>> print b[1].asnumpy()
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]

        >>> # pull a list of key-value pairs.
        >>> # On single device
        >>> keys = [5, 7, 9]
        >>> b = [mx.nd.zeros(shape)]*len(keys)
        >>> kv.pull(keys, out=b)
        >>> print b[1].asnumpy()
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]
        >>> # On multiple devices
        >>> b = [[mx.nd.ones(shape, gpu) for gpu in gpus]] * len(keys)
        >>> kv.pull(keys, out=b)
        >>> print b[1][1].asnumpy()
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]
        """
        assert(out is not None)
        ckeys, cvals = _ctype_key_value(key, out)
        check_call(_LIB.MXKVStorePull(
            self.handle, mx_uint(len(ckeys)), ckeys, cvals,
            ctypes.c_int(priority)))

    def set_optimizer(self, optimizer):
        """ Registers an optimizer with the store.

        When there are multiple machines, this operation (invoked from a worker node)
        will pack the optimizer and send it to all servers. It returns after
        this action is done.

        Parameters
        ----------
        optimizer : Optimizer
            the optimizer
        """
        is_worker = ctypes.c_int()
        check_call(_LIB.MXKVStoreIsWorkerNode(ctypes.byref(is_worker)))

        # pylint: disable=invalid-name
        if 'dist' in self.type and is_worker.value:
            # send the optimizer to server
            try:
                # use ASCII protocol 0, might be slower, but not a big ideal
                optim_str = pickle.dumps(optimizer, 0)
            except:
                raise
            self._send_command_to_servers(0, optim_str)
        else:
            self._set_updater(opt.get_updater(optimizer))

    def set_cancel_callback(self, callback):
        """Sets a callback for cancellation.
        """
        if 'dist' in self.type:
            proto = ctypes.CFUNCTYPE(None)
            check_call(_LIB.MXKVStoreSetCancelCallback(self.handle, proto(callback)))
        else:
            raise RuntimeError('set_cancel_callback is only supported in dist mode.')

    def report_loss(self, loss):
        if 'dist' in self.type:
            check_call(_LIB.MXKVStoreReportLoss(self.handle, ctypes.c_double(loss)))
        else:
            raise RuntimeError('report_loss is only supported in dist mode.')

    @property
    def type(self):
        """ Returns the type of this kvstore.

        Returns
        -------
        type : str
            the string type
        """
        kv_type = ctypes.c_char_p()
        check_call(_LIB.MXKVStoreGetType(self.handle, ctypes.byref(kv_type)))
        return py_str(kv_type.value)

    @property
    def rank(self):
        """ Returns the rank of this worker node.

        Returns
        -------
        rank : int
            The rank of this node, which is in range [0, num_workers())
        """
        rank = ctypes.c_int()
        check_call(_LIB.MXKVStoreGetRank(self.handle, ctypes.byref(rank)))
        return rank.value

    @property
    def num_workers(self):
        """Returns the number of worker nodes.

        Returns
        -------
        size :int
            The number of worker nodes.
        """
        size = ctypes.c_int()
        check_call(_LIB.MXKVStoreGetGroupSize(self.handle, ctypes.byref(size)))
        return size.value

    def save_optimizer_states(self, fname):
        """Saves optimizer (updater) state to file.

        Parameters
        ----------
        fname : str
            Path to output states file.
        """
        assert self._updater is not None, "Cannot save states for distributed training"
        with open(fname, 'wb') as fout:
            fout.write(self._updater.get_states())

    def load_optimizer_states(self, fname):
        """Loads optimizer (updater) state from file.

        Parameters
        ----------
        fname : str
            Path to input states file.
        """
        assert self._updater is not None, "Cannot save states for distributed training"
        self._updater.set_states(open(fname, 'rb').read())

    def _set_updater(self, updater):
        """Sets a push updater into the store.

        This function only changes the local store. When running on multiple machines one must
        use `set_optimizer`.

        Parameters
        ----------
        updater : function
            The updater function.

        Examples
        --------
        >>> def update(key, input, stored):
        ...     print "update on key: %d" % key
        ...     stored += input * 2
        >>> kv._set_updater(update)
        >>> kv.pull(3, out=a)
        >>> print a.asnumpy()
        [[ 4.  4.  4.]
        [ 4.  4.  4.]]
        >>> kv.push(3, mx.nd.ones(shape))
        update on key: 3
        >>> kv.pull(3, out=a)
        >>> print a.asnumpy()
        [[ 6.  6.  6.]
        [ 6.  6.  6.]]
        """
        self._updater = updater
        _updater_proto = ctypes.CFUNCTYPE(
            None, ctypes.c_int, NDArrayHandle, NDArrayHandle, ctypes.c_void_p)
        self._updater_func = _updater_proto(_updater_wrapper(updater))
        check_call(_LIB.MXKVStoreSetUpdater(self.handle, self._updater_func, None))

    def _barrier(self):
        """Invokes global barrier among all worker nodes.

        For example, assume there are `n` machines. We would like machine `0` to first
        `init` the values and then have all the workers `pull` the initialized value.
        Before pulling, we can place invoke `_barrier()` to guarantee that the
        initialization is finished.
        """
        check_call(_LIB.MXKVStoreBarrier(self.handle))

    def _send_command_to_servers(self, head, body):
        """Sends a command to all server nodes.

        Sending command to a server node will cause that server node to invoke
        ``KVStoreServer.controller`` to execute the command.

        This function returns after the command has been executed on all server
        nodes.

        Parameters
        ----------
        head : int
            the head of the command
        body : str
            the body of the command
        """
        check_call(_LIB.MXKVStoreSendCommmandToServers(
            self.handle, mx_uint(head), c_str(body)))

def create(name='local'):
    """Creates a new KVStore.

    Parameters
    ----------
    name : {'local'}
        The type of KVStore
        - local works for multiple devices on a single machine (single process).
        - dist works for multiple machines (multiple processes).
    Returns
    -------
    kv : KVStore
        The created KVStore.
    """
    if not isinstance(name, string_types):
        raise TypeError('name must be a string')
    handle = KVStoreHandle()
    check_call(_LIB.MXKVStoreCreate(c_str(name),
                                    ctypes.byref(handle)))
    return KVStore(handle)
