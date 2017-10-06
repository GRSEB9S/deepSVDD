import theano
import theano.tensor as T
import numpy as np

from utils import config_


def assign_empty_tensor(layer, name, ndim, on_gpu=True):

    tensor_name = layer.name + "_" + name
    setattr(layer, name, empty_shared_tensor(ndim, tensor_name, on_gpu))


def empty_shared_tensor(ndim, name=None, on_gpu=True):

    empty_array = np.empty([0] * ndim, dtype=config_.floatX)
    if on_gpu:
        return theano.shared(empty_array, name=name)
    else:
        return empty_array


def set_to_zero(list_of_tensors_and_shapes, on_gpu=True):
    """
    :param: list_of_tensors_and_shapes of the form [(tensor1, shape1), ...]
    """

    if on_gpu:
        updates = []
        for tensor, shape in list_of_tensors_and_shapes:
            if np.sum(shape) == 1:
                updates.append((tensor, 0))
            else:
                updates.append((tensor,
                                T.patternbroadcast(T.zeros(shape),
                                                   [False] * tensor.ndim)))

        return updates

    else:
        updates = []
        for tensor, shape in list_of_tensors_and_shapes:
            updates.append((tensor, np.zeros(shape, dtype=config_.floatX)))
        return updates


def deallocate_shared_tensor(tensor, on_gpu=True):

    empty_array = np.empty([0] * tensor.ndim, dtype=config_.floatX)

    if on_gpu:
        tensor.set_value(empty_array)
    else:
        tensor = empty_array


def is_zero(array):

    if hasattr(array, "get_value"):
        return np.linalg.norm(array.get_value()) < config_.eps

    return np.linalg.norm(array) < config_.eps


def batched_dot(A, B):
    """
    naive implementation of batched dot, for test purposes only
    """

    assert A.shape[0] == B.shape[0]

    lazy = np.array([np.dot(A[i], B[i]) for i in range(A.shape[0])])

    return lazy
