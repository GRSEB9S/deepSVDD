import theano
import theano.tensor as T

import lasagne

from utils.tensor import set_to_zero, deallocate_shared_tensor,\
    assign_empty_tensor
from config import Configuration as Cfg


def compile_hinge_avg(nnet, layer):

    X = T.tensor4("X")
    y = T.ivector("y")

    X_layer = lasagne.layers.get_output(layer.input_layer, X,
                                        deterministic=True)
    Z = layer.forward_with(X_layer, layer.W_avg, layer.b_avg)

    for next_layer in nnet.next_layers(layer):
        Z = next_layer.get_output_for(Z, deterministic=True)

    objective, acc = nnet.svm_layer.objective(Z, y)

    return theano.function([X, y], [objective, acc])


def compile_warm_reg_avg(layer):

    C, D = Cfg.C, Cfg.D
    r = C / (C + D)
    one_by_2K = Cfg.floatX(0.5) * (C + D) / (C * D)

    warm_reg = one_by_2K * (T.sum((layer.W_avg - r * layer.W_0) ** 2) +
                            T.sum((layer.b_avg - r * layer.b_0) ** 2))

    return theano.function([], warm_reg)


def warm_regularization(layer):

    C, D = Cfg.C, Cfg.D
    r = C / (C + D)
    one_by_2K = Cfg.floatX(0.5) * (C + D) / (C * D)

    warm_reg = one_by_2K * (T.sum((layer.W - r * layer.W_0) ** 2) +
                            T.sum((layer.b - r * layer.b_0) ** 2))

    return theano.function([], warm_reg)


def compile_norm_avg(layer):

    l2_norm = T.sum(layer.W_avg ** 2) + T.sum(layer.b_avg ** 2)
    return theano.function([], l2_norm)


def compute_norm(layer):

    l2_norm = T.sum(layer.W ** 2) + T.sum(layer.b ** 2)
    return theano.function([], l2_norm)


def update_fixed_weights(layer):

    updates = ((layer.W_fixed, layer.W),
               (layer.b_fixed, layer.b))

    return theano.function([], updates=updates)


def separate_weights(layer):

    assert layer.issvm or layer.isdense

    W_pos = Cfg.floatX(0.5) * (abs(layer.W) + layer.W)
    W_neg = Cfg.floatX(0.5) * (abs(layer.W) - layer.W)

    updates = ((layer.W_pos, W_pos),
               (layer.W_neg, W_neg))

    return theano.function([], updates=updates)


def separate_gamma(layer):

    assert layer.isbatchnorm

    gamma_pos = Cfg.floatX(0.5) * (abs(layer.gamma) + layer.gamma)
    gamma_neg = Cfg.floatX(0.5) * (abs(layer.gamma) - layer.gamma)

    updates = ((layer.gamma_pos, gamma_pos),
               (layer.gamma_neg, gamma_neg))

    return theano.function([], updates=updates)


def use_average(layer):

    updates = ((layer.W, layer.W_avg),
               (layer.b, layer.b_avg))

    return theano.function([], updates=updates)


def get_dual(layer):

    C, D = Cfg.C, Cfg.D
    r = C / (C + D)
    K = C * D / (C + D)

    dual = Cfg.floatX(-0.5) / K * T.sum((layer.W - r * layer.W_0)**2) + \
        layer.l + r / K * T.sum(layer.W_0 * (r * layer.W_0 - layer.W))

    return theano.function([], dual)


def initialize_warm_start(layer):

    W_0 = layer.W
    b_0 = layer.b

    r = Cfg.C / (Cfg.C + Cfg.D)
    W = r * W_0
    b = r * b_0

    updates = ((layer.W, W),
               (layer.b, b),
               (layer.W_avg, W),
               (layer.b_avg, b),
               (layer.W_0, W_0),
               (layer.b_0, b_0))

    return theano.function([], updates=updates)


def input_batch(layer):

    idx = T.iscalar()
    X = T.tensor4()

    layer_input = lasagne.layers.get_output(layer.input_layer, X,
                                            deterministic=True)
    layer_input = layer_input.flatten(2) if layer_input.ndim > layer.inp_ndim \
        else layer_input

    b_size = X.shape[0]
    X_layer = T.set_subtensor(layer.X_layer[idx, :b_size, :], layer_input)

    updates = [(layer.X_layer, X_layer)]

    return theano.function([idx, X], updates=updates)


def get_input_for(layer):

    def my_fun(X, deterministic=True):

        layer_input = lasagne.layers.get_output(layer.input_layer, X,
                                                deterministic=deterministic)

        if (layer.isdense and layer_input.ndim > 2):
            layer_input = layer_input.flatten(2)

        return layer_input

    return my_fun


def deallocate_primal_variables_svm(layer):

    def deallocation_fun():

        deallocate_shared_tensor(layer.W_0)
        deallocate_shared_tensor(layer.b_0)

        deallocate_shared_tensor(layer.W_i, on_gpu=Cfg.store_on_gpu)
        deallocate_shared_tensor(layer.b_i, on_gpu=Cfg.store_on_gpu)
        deallocate_shared_tensor(layer.l_i, on_gpu=Cfg.store_on_gpu)

        if Cfg.store_on_gpu:
            deallocate_shared_tensor(layer.X_layer)

    return deallocation_fun


def deallocate_primal_variables_dense(layer):

    def deallocation_fun():

        deallocate_shared_tensor(layer.W_0)
        deallocate_shared_tensor(layer.b_0)

        deallocate_shared_tensor(layer.W_fixed)
        deallocate_shared_tensor(layer.b_fixed)

        deallocate_shared_tensor(layer.W_i, on_gpu=Cfg.store_on_gpu)
        deallocate_shared_tensor(layer.b_i, on_gpu=Cfg.store_on_gpu)
        deallocate_shared_tensor(layer.l_i, on_gpu=Cfg.store_on_gpu)

        if Cfg.store_on_gpu:
            deallocate_shared_tensor(layer.X_layer)

    return deallocation_fun


def deallocate_primal_variables_conv(layer):

    def deallocation_fun():

        deallocate_shared_tensor(layer.W_0)
        deallocate_shared_tensor(layer.b_0)

        deallocate_shared_tensor(layer.W_fixed)
        deallocate_shared_tensor(layer.b_fixed)

        deallocate_shared_tensor(layer.W_i, on_gpu=Cfg.store_on_gpu)
        deallocate_shared_tensor(layer.b_i, on_gpu=Cfg.store_on_gpu)
        deallocate_shared_tensor(layer.l_i, on_gpu=Cfg.store_on_gpu)

    return deallocation_fun


def initialize_variables_conv(layer):

    layer.W_shape = list(layer.W.get_value().shape)
    layer.b_shape = list(layer.b.get_value().shape)

    assign_empty_tensor(layer, "W_0", 4)
    assign_empty_tensor(layer, "b_0", 1)

    assign_empty_tensor(layer, "W_avg", 4)
    assign_empty_tensor(layer, "b_avg", 1)

    assign_empty_tensor(layer, "W_fixed", 4)
    assign_empty_tensor(layer, "b_fixed", 1)

    assign_empty_tensor(layer, "dW", 4)
    assign_empty_tensor(layer, "db", 1)

    assign_empty_tensor(layer, "loss", 0)
    assign_empty_tensor(layer, "k", 0)
    assign_empty_tensor(layer, "l", 0)
    assign_empty_tensor(layer, "gamma", 0)

    assign_empty_tensor(layer, "W_i", 5, on_gpu=Cfg.store_on_gpu)
    assign_empty_tensor(layer, "b_i", 2, on_gpu=Cfg.store_on_gpu)
    assign_empty_tensor(layer, "l_i", 1, on_gpu=Cfg.store_on_gpu)

    if not Cfg.store_on_gpu:
        assign_empty_tensor(layer, "W_i_buffer", 4)
        assign_empty_tensor(layer, "b_i_buffer", 1)
        assign_empty_tensor(layer, "l_i_buffer", 0)


def initialize_variables_batchnorm(layer):

    if layer.input_layer.isdense:
        assign_empty_tensor(layer, "gamma_pos", 1)
        assign_empty_tensor(layer, "gamma_neg", 1)


def initialize_variables_dense(layer):

    layer.W_shape = list(layer.W.get_value().shape)
    layer.b_shape = list(layer.b.get_value().shape)

    assign_empty_tensor(layer, "W_0", 2)
    assign_empty_tensor(layer, "b_0", 1)

    assign_empty_tensor(layer, "W_pos", 2)
    assign_empty_tensor(layer, "W_neg", 2)

    assign_empty_tensor(layer, "W_fixed", 2)
    assign_empty_tensor(layer, "b_fixed", 1)

    assign_empty_tensor(layer, "W_avg", 2)
    assign_empty_tensor(layer, "b_avg", 1)

    assign_empty_tensor(layer, "dW", 2)
    assign_empty_tensor(layer, "db", 1)

    assign_empty_tensor(layer, "loss", 0)
    assign_empty_tensor(layer, "k", 0)
    assign_empty_tensor(layer, "l", 0)
    assign_empty_tensor(layer, "gamma", 0)

    assign_empty_tensor(layer, "W_i", 3, on_gpu=Cfg.store_on_gpu)
    assign_empty_tensor(layer, "b_i", 2, on_gpu=Cfg.store_on_gpu)
    assign_empty_tensor(layer, "l_i", 1, on_gpu=Cfg.store_on_gpu)

    if Cfg.store_on_gpu:
        assign_empty_tensor(layer, "X_layer", 3)
    else:
        assign_empty_tensor(layer, "W_i_buffer", 2)
        assign_empty_tensor(layer, "b_i_buffer", 1)
        assign_empty_tensor(layer, "l_i_buffer", 0)


def initialize_variables_svm(layer):

    layer.n_classes = layer.num_units
    layer.W_shape = list(layer.W.get_value().shape)
    layer.b_shape = list(layer.b.get_value().shape)

    layer.use_dc = False

    assign_empty_tensor(layer, "W_pos", 2)
    assign_empty_tensor(layer, "W_neg", 2)

    assign_empty_tensor(layer, "W_0", 2)
    assign_empty_tensor(layer, "b_0", 1)

    assign_empty_tensor(layer, "W_avg", 2)
    assign_empty_tensor(layer, "b_avg", 1)

    assign_empty_tensor(layer, "dW", 2)
    assign_empty_tensor(layer, "db", 1)

    assign_empty_tensor(layer, "loss", 0)
    assign_empty_tensor(layer, "k", 0)
    assign_empty_tensor(layer, "l", 0)
    assign_empty_tensor(layer, "gamma", 0)

    assign_empty_tensor(layer, "W_i", 3, on_gpu=Cfg.store_on_gpu)
    assign_empty_tensor(layer, "b_i", 2, on_gpu=Cfg.store_on_gpu)
    assign_empty_tensor(layer, "l_i", 1, on_gpu=Cfg.store_on_gpu)

    if Cfg.store_on_gpu:
        assign_empty_tensor(layer, "X_layer", 3)
    else:
        assign_empty_tensor(layer, "W_i_buffer", 2)
        assign_empty_tensor(layer, "b_i_buffer", 1)
        assign_empty_tensor(layer, "l_i_buffer", 0)


def update_cpu_fun(cpu_vars, layer):

    def update_fun():

        if len(cpu_vars):
            updates = set_to_zero(cpu_vars, on_gpu=False)
            layer.W_i = updates[0][1]
            layer.b_i = updates[1][1]
            layer.l_i = updates[2][1]
        else:
            pass

    return update_fun


def update_gpu_fun(gpu_vars):

    updates = set_to_zero(gpu_vars, on_gpu=True)

    return theano.function([], updates=updates)


def allocate_primal_variables_conv(layer):
    """ allocate variables on GPU / RAM for conv layers
    """

    assert layer.isconv

    gpu_vars = ((layer.l, [1]),
                (layer.k, [1]))
    cpu_vars = ()

    heavy_vars = ((layer.W_i, [Cfg.n_batches] + layer.W_shape),
                  (layer.b_i, [Cfg.n_batches] + layer.b_shape),
                  (layer.l_i, [Cfg.n_batches]))

    if Cfg.store_on_gpu:
        gpu_vars += heavy_vars
    else:
        cpu_vars += heavy_vars

    gpu_fun = update_gpu_fun(gpu_vars)
    cpu_fun = update_cpu_fun(cpu_vars, layer)

    def update_all_fun():
        gpu_fun()
        cpu_fun()

    return update_all_fun


def allocate_primal_variables_dense(layer):

    assert layer.isdense

    gpu_vars = ((layer.l, [1]),
                (layer.k, [1]))
    cpu_vars = ()

    heavy_vars = ((layer.W_i,
                   [Cfg.n_batches, Cfg.batch_size] + layer.b_shape),
                  (layer.b_i,
                   [Cfg.n_batches] + layer.b_shape),
                  (layer.l_i,
                   [Cfg.n_batches]))

    if Cfg.store_on_gpu:
        gpu_vars += heavy_vars
        gpu_vars += ((layer.X_layer,
                      [Cfg.n_batches, Cfg.batch_size, layer.W_shape[0]]),)
    else:
        cpu_vars += heavy_vars

    gpu_fun = update_gpu_fun(gpu_vars)
    cpu_fun = update_cpu_fun(cpu_vars, layer)

    def update_all_fun():
        gpu_fun()
        cpu_fun()

    return update_all_fun


def allocate_primal_variables_svm(layer):

    assert layer.issvm

    gpu_vars = ((layer.l, [1]),
                (layer.k, [1]))
    cpu_vars = ()

    # compress feature vectors of SVM if not storing on GPU
    if Cfg.store_on_gpu:
        heavy_vars = ((layer.W_i,
                       [Cfg.n_batches] + layer.W_shape),
                      (layer.b_i,
                       [Cfg.n_batches] + layer.b_shape),
                      (layer.l_i,
                       [Cfg.n_batches]))
    else:
        heavy_vars = ((layer.W_i,
                       [Cfg.n_batches, Cfg.batch_size] + layer.b_shape),
                      (layer.b_i,
                       [Cfg.n_batches] + layer.b_shape),
                      (layer.l_i,
                       [Cfg.n_batches]))

    if Cfg.store_on_gpu:
        gpu_vars += heavy_vars
        gpu_vars += ((layer.X_layer,
                      [Cfg.n_batches, Cfg.batch_size, layer.W_shape[0]]),)
    else:
        cpu_vars += heavy_vars

    gpu_fun = update_gpu_fun(gpu_vars)
    cpu_fun = update_cpu_fun(cpu_vars, layer)

    def update_all_fun():
        gpu_fun()
        cpu_fun()

    return update_all_fun


def reinitialize_primal_variables(layer):

    gpu_vars = (layer.l,
                layer.k)
    cpu_vars = ()

    heavy_vars = (layer.W_i,
                  layer.b_i,
                  layer.l_i)

    if Cfg.store_on_gpu:
        gpu_vars += heavy_vars
    else:
        cpu_vars += heavy_vars

    zero = Cfg.floatX(0)

    gpu_updates = []
    for var in gpu_vars:
        gpu_updates.append((var, var.fill(zero)))
    gpu_fun = theano.function([], updates=gpu_updates)

    def update_all_fun():
        gpu_fun()
        for var in cpu_vars:
            var.fill(zero)

    return update_all_fun


def precompute_input(layer, nnet):

    assert layer.isdense or layer.issvm
    assert Cfg.store_on_gpu, \
        "should not precompute inputs when memory requirements are high"

    for batch in nnet.data.get_epoch_train():
        X, _, idx = batch
        layer.get_input_batch(idx, X)


def prepare_dc_decomposition(nnet, layer):

    assert layer.isdense, "only implemented for dense layers for now"

    # all following layers should be separated
    for next_layer in nnet.next_layers(layer):
        if next_layer.isdense or next_layer.issvm:
            next_layer.separate_weights()
        if next_layer.isbatchnorm:
            next_layer.separate_gamma()
