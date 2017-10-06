import theano
import theano.tensor as T
import numpy as np
import lasagne.layers
import lasagne.nonlinearities
import lasagne.updates as l_updates
import lasagne.objectives as l_objectives

from lasagne.regularization import regularize_network_params, l2
from theano import shared
from config import Configuration as Cfg
# from theano.compile.nanguardmode import NanGuardMode


def get_updates(nnet,
                train_obj,
                trainable_params,
                solver=None):

    implemented_solvers = ("sgd", "momentum", "nesterov", "adagrad", "rmsprop",
                           "adadelta", "adam", "adamax")

    if solver not in implemented_solvers:
        nnet.sgd_solver = "nesterov"
    else:
        nnet.sgd_solver = solver

    if nnet.sgd_solver == "sgd":
        updates = l_updates.sgd(train_obj,
                                trainable_params,
                                learning_rate=Cfg.learning_rate)
    elif nnet.sgd_solver == "momentum":
        updates = l_updates.momentum(train_obj,
                                     trainable_params,
                                     learning_rate=Cfg.learning_rate,
                                     momentum=Cfg.momentum)
    elif nnet.sgd_solver == "nesterov":
        updates = l_updates.nesterov_momentum(train_obj,
                                              trainable_params,
                                              learning_rate=Cfg.learning_rate,
                                              momentum=Cfg.momentum)
    elif nnet.sgd_solver == "adagrad":
        updates = l_updates.adagrad(train_obj,
                                    trainable_params,
                                    learning_rate=Cfg.learning_rate)
    elif nnet.sgd_solver == "rmsprop":
        updates = l_updates.rmsprop(train_obj,
                                    trainable_params,
                                    learning_rate=Cfg.learning_rate,
                                    rho=Cfg.rho)
    elif nnet.sgd_solver == "adadelta":
        updates = l_updates.adadelta(train_obj,
                                     trainable_params,
                                     learning_rate=Cfg.learning_rate,
                                     rho=Cfg.rho)
    elif nnet.sgd_solver == "adam":
        updates = l_updates.adam(train_obj,
                                 trainable_params,
                                 learning_rate=Cfg.learning_rate)
    elif nnet.sgd_solver == "adamax":
        updates = l_updates.adamax(train_obj,
                                   trainable_params,
                                   learning_rate=Cfg.learning_rate)

    return updates


def get_l2_penalty(nnet, include_bias=False, pow=2):
    """
    returns the l2 penalty on (trainable) network parameters combined as sum
    """

    l2_penalty = 0

    # do not include OC-SVM layer in regularization
    if Cfg.ocsvm_loss:
        if include_bias:
            for layer in nnet.trainable_layers:
                if not layer.issvm:
                    if layer.b is not None:
                        l2_penalty = (l2_penalty + T.sum(abs(layer.W) ** pow)
                                      + T.sum(abs(layer.b) ** pow))
                    else:
                        l2_penalty = l2_penalty + T.sum(abs(layer.W) ** pow)
        else:
            for layer in nnet.trainable_layers:
                if not layer.issvm:
                    l2_penalty = l2_penalty + T.sum(abs(layer.W) ** pow)
    else:
        if include_bias:
            for layer in nnet.trainable_layers:
                if layer.b is not None:
                    l2_penalty = (l2_penalty + T.sum(abs(layer.W) ** pow)
                                  + T.sum(abs(layer.b) ** pow))
                else:
                    l2_penalty = l2_penalty + T.sum(abs(layer.W) ** pow)
        else:
            for layer in nnet.trainable_layers:
                l2_penalty = l2_penalty + T.sum(abs(layer.W) ** pow)

    return T.cast(l2_penalty, dtype='floatX')


def get_sparsity_penalty(nnet, inputs, sparsity, mode="mean",
                         deterministic=False):
    """
    returns the sparsity penalty on network activations combined as a sum
    """

    assert mode in ("mean", "l1")

    rho = sparsity
    penalty = 0
    eps = 0.0001  # for numerical stability

    for layer in nnet.all_layers:
        if layer.isactivation:

            activation = lasagne.layers.get_output(layer, inputs=inputs,
                                                   deterministic=deterministic)

            if mode == "mean":
                if layer.isrelu:
                    avg_activation = T.mean(T.gt(activation,
                                                 T.zeros_like(activation)),
                                            axis=0, dtype='floatX')
                if layer.issigmoid:
                    avg_activation = T.mean(activation, axis=0, dtype='floatX')

                KL_div = T.sum((rho+eps) *
                               (T.log(rho+eps) - T.log(avg_activation+eps)) +
                               (1-rho+eps) *
                               (T.log(1-rho+eps) - T.log(1-avg_activation+eps)),
                               dtype='floatX')
                penalty = penalty + KL_div

            if mode == "l1":
                penalty = penalty + T.sum(abs(activation), dtype='floatX')

    return T.cast(penalty, dtype='floatX')


def get_spectral_penalty(nnet, include_bias=False):
    """
    returns the sum of squared spectral norms of network parameters
    (i.e. the sum of the largest Eigenvalues of dot(W, W.T))
    """

    penalty = 0

    for layer in nnet.trainable_layers:
        if not layer.issvm:
            eigenvalues, eigvec = T.nlinalg.eigh(T.dot(layer.W, layer.W.T))
            eig_max = T.max(eigenvalues)
            penalty = penalty + eig_max

    if include_bias:
        for layer in nnet.trainable_layers:
            if (not layer.issvm) and (layer.b is not None):
                penalty = penalty + T.sum(abs(layer.b) ** 2)

    return T.cast(penalty, dtype='floatX')


def get_prod_penalty(nnet):
    """
    returns the l2 penalty on (trainable) network parameters combined as tensor
    product.
    """

    assert Cfg.ocsvm_loss is True

    penalty = 0
    layers = nnet.trainable_layers
    num_layers = len(layers) - 1  # do not regularize parameters of oc-svm layer
    assert num_layers > 0

    W_norm_prod = 1.0

    if layers[num_layers-1].b is not None:
        penalty += T.sum(layers[num_layers-1].b ** 2)

    for i in range(num_layers-1):
        W_norm_prod *= T.sum(layers[num_layers-1-i].W ** 2)
        if layers[num_layers-2-i].b is not None:
            penalty += W_norm_prod * T.sum(layers[num_layers-2-i].b ** 2)

    W_norm_prod *= T.sum(layers[0].W ** 2)

    penalty += W_norm_prod
    penalty *= T.sum(nnet.ocsvm_layer.W ** 2)

    return penalty


def get_bias_offset(nnet):
    """
    returns the offset to balance the polynomial parameters possible by the bias
    terms of the network.
    """

    offset = 0
    L = len(nnet.trainable_layers)

    for l in range(L-1):
        layer = nnet.trainable_layers[l]

        if layer.b is not None:
            W_prod = T.eye(int(layer.b.shape.eval()[0]))

            for k in range(1, L-1):
                W_prod = T.dot(nnet.trainable_layers[k].W.T, W_prod)

            offset = offset + T.dot(W_prod, layer.b)

    offset = T.dot(nnet.ocsvm_layer.W.T, offset)

    return T.sum(offset)


def create_update(nnet):
    """ create update for network given in argument
    """

    if nnet.data._X_val.ndim == 2:
        inputs = T.matrix('inputs')
    elif nnet.data._X_val.ndim == 4:
        inputs = T.tensor4('inputs')

    targets = T.ivector('targets')

    # compile theano functions
    if Cfg.softmax_loss:
        compile_update_softmax(nnet, inputs, targets)
    elif Cfg.ocsvm_loss:
        if Cfg.rho_fixed:
            compile_update_ocsvm_rho_fixed(nnet, inputs, targets)
        else:
            compile_update_ocsvm(nnet, inputs, targets)
    elif Cfg.svdd_loss:
        compile_update_svdd(nnet, inputs, targets)
    elif Cfg.reconstruction_loss:
        create_autoencoder(nnet)
    else:
        compile_update_default(nnet, inputs, targets)


def compile_update_default(nnet, inputs, targets):
    """
    create a SVM loss for network given in argument
    """

    floatX = Cfg.floatX
    C = Cfg.C

    if len(nnet.all_layers) > 1:
        feature_layer = nnet.all_layers[-2]
    else:
        feature_layer = nnet.input_layer
    final_layer = nnet.svm_layer
    trainable_params = lasagne.layers.get_all_params(final_layer,
                                                     trainable=True)

    # Regularization
    if Cfg.weight_decay:
        l2_penalty = (floatX(0.5) / C) * get_l2_penalty(nnet, Cfg.include_bias)
    else:
        l2_penalty = T.cast(0, dtype='floatX')

    # Backpropagation
    prediction = lasagne.layers.get_output(final_layer, inputs=inputs,
                                           deterministic=False)
    objective, train_acc = final_layer.objective(prediction, targets)
    train_loss = T.cast((objective) / targets.shape[0], dtype='floatX')
    train_acc = T.cast(train_acc * 1. / targets.shape[0], dtype='floatX')
    train_obj = l2_penalty + train_loss
    updates = get_updates(nnet, train_obj, trainable_params, solver=nnet.solver)
    nnet.backprop = theano.function([inputs, targets],
                                    [train_obj, train_acc],
                                    updates=updates)

    # Hinge loss
    nnet.hinge_loss = theano.function([inputs, targets],
                                      [train_loss, train_acc])

    # Forwardpropagation
    test_prediction = lasagne.layers.get_output(final_layer, inputs=inputs,
                                                deterministic=True)
    if nnet.data.n_classes == 2:
        scores = test_prediction[:, 1] - test_prediction[:, 0]
    else:
        scores = T.zeros_like(targets)
    objective, test_acc = final_layer.objective(test_prediction, targets)
    test_loss = T.cast(objective / targets.shape[0], dtype='floatX')
    test_acc = T.cast(test_acc * 1. / targets.shape[0], dtype='floatX')
    test_obj = l2_penalty + test_loss
    # get network feature representation
    test_rep = lasagne.layers.get_output(feature_layer, inputs=inputs,
                                         deterministic=True)
    test_rep_norm = test_rep.norm(L=2, axis=1)
    nnet.forward = theano.function([inputs, targets],
                                   [test_obj, test_acc, scores, l2_penalty,
                                    test_rep_norm, test_loss])


def compile_update_softmax(nnet, inputs, targets):
    """
    create a softmax loss for network given in argument
    """

    floatX = Cfg.floatX
    C = Cfg.C

    final_layer = nnet.all_layers[-1]
    trainable_params = lasagne.layers.get_all_params(final_layer,
                                                     trainable=True)

    # Regularization
    if Cfg.weight_decay:
        l2_penalty = (floatX(0.5) / C) * get_l2_penalty(nnet, Cfg.include_bias)
    else:
        l2_penalty = T.cast(0, dtype='floatX')

    # Backpropagation
    prediction = lasagne.layers.get_output(final_layer, inputs=inputs,
                                           deterministic=False)
    if Cfg.ad_experiment:
        train_loss = T.mean(l_objectives.binary_crossentropy(
            prediction.flatten(), targets),
            dtype='floatX')
        train_acc = T.mean(l_objectives.binary_accuracy(prediction.flatten(),
                                                        targets),
                           dtype='floatX')
    else:
        train_loss = T.mean(l_objectives.categorical_crossentropy(prediction,
                                                                  targets),
                            dtype='floatX')
        train_acc = T.mean(T.eq(T.argmax(prediction, axis=1), targets),
                           dtype='floatX')


    train_obj = T.cast(train_loss + l2_penalty, dtype='floatX')
    updates = get_updates(nnet, train_obj, trainable_params, solver=nnet.solver)
    nnet.backprop = theano.function([inputs, targets],
                                    [train_obj, train_acc],
                                    updates=updates)

    # Forwardpropagation
    test_prediction = lasagne.layers.get_output(final_layer, inputs=inputs,
                                                deterministic=True)
    if Cfg.ad_experiment:
        test_loss = T.mean(l_objectives.binary_crossentropy(
            test_prediction.flatten(), targets), dtype='floatX')
        test_acc = T.mean(l_objectives.binary_accuracy(
            test_prediction.flatten(), targets), dtype='floatX')
    else:
        test_loss = T.mean(l_objectives.categorical_crossentropy(
            test_prediction, targets), dtype='floatX')
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), targets),
               dtype='floatX')
    test_obj = T.cast(test_loss + l2_penalty, dtype='floatX')
    nnet.forward = theano.function([inputs, targets],
                                   [test_obj, test_acc, test_prediction,
                                    l2_penalty, test_loss])


def compile_update_ocsvm(nnet, inputs, targets):
    """
    create a OC-SVM loss for network given in argument
    """

    floatX = Cfg.floatX
    C = Cfg.C
    A = Cfg.A
    nu = Cfg.nu

    if len(nnet.all_layers) > 1:
        feature_layer = nnet.all_layers[-2]
    else:
        feature_layer = nnet.input_layer
    final_layer = nnet.ocsvm_layer
    trainable_params = lasagne.layers.get_all_params(final_layer,
                                                     trainable=True)

    # Regularization (up to feature map)
    if Cfg.weight_decay:
        if Cfg.prod_penalty:
            l2_penalty = (1/C) * get_prod_penalty(nnet)
        elif Cfg.spec_penalty:
            l2_penalty = (1/C) * get_spectral_penalty(nnet, Cfg.include_bias)
        else:
            l2_penalty = ((1/C) * get_l2_penalty(nnet,
                                                 include_bias=Cfg.include_bias,
                                                 pow=Cfg.pow))
    else:
        l2_penalty = T.cast(0, dtype='floatX')

    # Bias offset
    if Cfg.bias_offset:
        bias_offset = get_bias_offset(nnet)
    else:
        bias_offset = T.cast(0, dtype='floatX')

    # Backpropagation
    prediction = lasagne.layers.get_output(final_layer, inputs=inputs,
                                           deterministic=False)
    objective, train_acc = final_layer.objective(prediction, targets)

    # Normalization
    rep = lasagne.layers.get_output(feature_layer, inputs=inputs,
                                    deterministic=False)
    rep_norm = rep.norm(L=2, axis=1).dimshuffle((0, 'x'))
    if Cfg.ball_penalty:
        ball_penalty, _ = final_layer.objective(
            T.ones_like(rep_norm) - (rep_norm ** 2), targets)
    else:
        ball_penalty = T.cast(0, dtype='floatX')
    ball_penalty = (1/A) * T.cast(ball_penalty / targets.shape[0],
                                  dtype='floatX')

    # Output regularization
    if Cfg.output_penalty:
        l2_output = (1/C) * (T.sum(abs(final_layer.W) ** Cfg.pow)
                             * T.sum(abs(rep) ** 2))
    else:
        l2_output = T.cast(0, dtype='floatX')
    l2_output = T.cast(l2_output / targets.shape[0], dtype='floatX')

    # SVM parameter regularization
    if Cfg.Wsvm_penalty:
        Wsvm_penalty = T.sum(abs(final_layer.W) ** Cfg.pow)
    else:
        Wsvm_penalty = T.cast(0, dtype='floatX')

    # OC SVM loss has nu parameter and adds margin from origin to objective
    train_loss = T.cast(objective / (targets.shape[0] * nu), dtype='floatX')
    train_acc = T.cast(train_acc * 1. / targets.shape[0], dtype='floatX')
    train_obj = T.cast(floatX(0.5) * l2_penalty
                       + floatX(0.5) * ball_penalty
                       + floatX(0.5) * l2_output
                       + floatX(0.5) * Wsvm_penalty
                       + train_loss
                       + T.sum(final_layer.b)
                       + bias_offset, dtype='floatX')
    updates = get_updates(nnet, train_obj, trainable_params, solver=nnet.solver)
    nnet.backprop = theano.function([inputs, targets],
                                    [train_obj, train_acc],
                                    updates=updates)

    # Forwardpropagation
    test_prediction = lasagne.layers.get_output(final_layer, inputs=inputs,
                                                deterministic=True)
    # get network feature representation
    test_rep = lasagne.layers.get_output(feature_layer, inputs=inputs,
                                         deterministic=True)
    test_rep_norm = test_rep.norm(L=2, axis=1)
    if Cfg.ball_penalty:
        test_ball_penalty, _ = final_layer.objective(
            T.ones_like(test_rep_norm.dimshuffle((0, 'x')))
            - (test_rep_norm.dimshuffle((0, 'x')) ** 2), targets)
    else:
        test_ball_penalty = T.cast(0, dtype='floatX')
    test_ball_penalty = ((1/A) * T.cast(
        test_ball_penalty / targets.shape[0], dtype='floatX'))

    # Output regularization
    if Cfg.output_penalty:
        test_l2_output = (1/C) * (T.sum(abs(final_layer.W) ** Cfg.pow)
                                  * T.sum(abs(test_rep) ** 2))
    else:
        test_l2_output = T.cast(0, dtype='floatX')
    test_l2_output = T.cast(test_l2_output / targets.shape[0], dtype='floatX')

    objective, test_acc = final_layer.objective(test_prediction, targets)
    test_loss = T.cast(objective / (targets.shape[0] * nu), dtype='floatX')
    test_acc = T.cast(test_acc * 1. / targets.shape[0], dtype='floatX')
    test_obj = T.cast(floatX(0.5) * l2_penalty
                      + floatX(0.5) * test_ball_penalty
                      + floatX(0.5) * test_l2_output
                      + floatX(0.5) * Wsvm_penalty
                      + test_loss
                      + T.sum(final_layer.b), dtype='floatX')
    nnet.forward = theano.function([inputs, targets],
                                   [test_obj, test_acc, test_prediction,
                                    floatX(0.5) * l2_penalty,
                                    floatX(0.5) * test_l2_output, test_rep,
                                    test_rep_norm, test_loss,
                                    floatX(0.5) * test_ball_penalty])


def compile_update_ocsvm_rho_fixed(nnet, inputs, targets):
    """
    create a OC-SVM loss for network given in argument with rho=1 fixed
    """

    floatX = Cfg.floatX
    C = Cfg.C
    nu = Cfg.nu

    if len(nnet.all_layers) > 1:
        feature_layer = nnet.all_layers[-2]
    else:
        feature_layer = nnet.input_layer
    final_layer = nnet.ocsvm_layer
    trainable_params = lasagne.layers.get_all_params(final_layer,
                                                     trainable=True)

    # Regularization
    Wsvm_penalty = T.sum(abs(final_layer.W) ** Cfg.pow)

    l2_penalty = get_l2_penalty(nnet,
                                include_bias=Cfg.include_bias,
                                pow=Cfg.pow)
    l2_penalty += Wsvm_penalty
    l2_penalty *= (1/C)

    # Backpropagation
    prediction = lasagne.layers.get_output(final_layer, inputs=inputs,
                                           deterministic=False)
    scores = T.ones_like(prediction) - prediction
    objective, train_acc = final_layer.objective(-scores, targets)

    # OC-SVM loss
    train_loss = T.cast(objective / (targets.shape[0] * nu), dtype='floatX')
    train_acc = T.cast(train_acc * 1. / targets.shape[0], dtype='floatX')
    train_obj = T.cast(floatX(0.5) * l2_penalty + train_loss, dtype='floatX')
    updates = get_updates(nnet, train_obj, trainable_params, solver=nnet.solver)
    nnet.backprop = theano.function([inputs, targets],
                                    [train_obj, train_acc],
                                    updates=updates)

    # Forwardpropagation
    test_prediction = lasagne.layers.get_output(final_layer, inputs=inputs,
                                                deterministic=True)
    test_scores = T.ones_like(prediction) - test_prediction
    objective, test_acc = final_layer.objective(-test_scores, targets)

    # Get network feature representation
    test_rep = lasagne.layers.get_output(feature_layer, inputs=inputs,
                                         deterministic=True)
    test_rep_norm = test_rep.norm(L=2, axis=1)

    test_ball_penalty = T.cast(0, dtype='floatX')
    test_l2_output = T.cast(0, dtype='floatX')

    # OC-SVM test loss
    test_loss = T.cast(objective / (targets.shape[0] * nu), dtype='floatX')
    test_acc = T.cast(test_acc * 1. / targets.shape[0], dtype='floatX')
    test_obj = T.cast(floatX(0.5) * l2_penalty + test_loss, dtype='floatX')
    nnet.forward = theano.function([inputs, targets],
                                   [test_obj, test_acc, -test_scores,
                                    floatX(0.5) * l2_penalty,
                                    floatX(0.5) * test_l2_output, test_rep,
                                    test_rep_norm, test_loss,
                                    floatX(0.5) * test_ball_penalty])


def compile_update_svdd(nnet, inputs, targets):
    """
    create a SVDD loss for network given in argument
    """

    floatX = Cfg.floatX
    B = Cfg.B
    C = Cfg.C
    nu = Cfg.nu

    # initialize R
    if nnet.R_init > 0:
        nnet.Rvar = shared(floatX(nnet.R_init), name="R")
    else:
        nnet.Rvar = shared(floatX(1), name="R")  # initialization with R=1

    # Loss
    feature_layer = nnet.all_layers[-1]
    rep = lasagne.layers.get_output(feature_layer, inputs=inputs,
                                    deterministic=False)

    # initialize c (0.5 in every feature representation dimension)
    rep_dim = feature_layer.num_units
    # nnet.cvar = shared(floatX(np.ones(rep_dim) * (1. / (rep_dim ** 0.5))),
    #                    name="c")
    nnet.cvar = shared(floatX(np.ones(rep_dim) * 0.5), name="c")

    dist = T.sum(((rep - nnet.cvar.dimshuffle('x', 0)) ** 2),
                 axis=1, dtype='floatX')
    scores = dist - nnet.Rvar
    stack = T.stack([T.zeros_like(scores), scores], axis=1)
    loss = T.cast(T.sum(T.max(stack, axis=1)) / (inputs.shape[0] * nu),
                  dtype='floatX')

    y_pred = T.argmax(stack, axis=1)
    acc = T.cast((T.sum(T.eq(y_pred.flatten(), targets), dtype='int32')
                  * 1. / targets.shape[0]), 'floatX')

    # Network weight decay
    if Cfg.weight_decay:
        l2_penalty = (1/C) * get_l2_penalty(nnet,
                                            include_bias=Cfg.include_bias,
                                            pow=Cfg.pow)
    else:
        l2_penalty = T.cast(0, dtype='floatX')

    # Network activation sparsity regularization
    if Cfg.sparsity_penalty:
        sparsity_penalty = (1/B) * get_sparsity_penalty(nnet, inputs,
                                                        Cfg.sparsity,
                                                        mode=Cfg.sparsity_mode,
                                                        deterministic=False)
    else:
        sparsity_penalty = T.cast(0, dtype='floatX')

    # Backpropagation (hard-margin: only minimizing everything to a ball
    # centered at c)
    trainable_params = lasagne.layers.get_all_params(feature_layer,
                                                     trainable=True)
    if Cfg.gaussian_blob:
        avg_dist = T.mean(1-T.exp(-dist), dtype="floatX")
    else:
        avg_dist = T.mean(dist, dtype="floatX")
    obj_ball = T.cast(floatX(0.5) * l2_penalty + avg_dist + sparsity_penalty,
                      dtype='floatX')
    updates_ball = get_updates(nnet, obj_ball, trainable_params,
                               solver=nnet.solver)
    nnet.backprop_ball = theano.function([inputs, targets], [obj_ball, acc],
                                         updates=updates_ball)

    # Backpropagation (without training R)
    obj = T.cast(floatX(0.5) * l2_penalty + nnet.Rvar + loss + sparsity_penalty,
                 dtype='floatX')
    updates = get_updates(nnet, obj, trainable_params, solver=nnet.solver)
    nnet.backprop_without_R = theano.function([inputs, targets], [obj, acc],
                                              updates=updates)

    # Backpropagation (with training R)
    trainable_params.append(nnet.Rvar)  # add radius R to trainable parameters
    updates = get_updates(nnet, obj, trainable_params, solver=nnet.solver)
    nnet.backprop = theano.function([inputs, targets], [obj, acc],
                                    updates=updates)


    # Forwardpropagation
    test_rep = lasagne.layers.get_output(feature_layer, inputs=inputs,
                                         deterministic=True)
    test_rep_norm = test_rep.norm(L=2, axis=1)

    test_dist = T.sum(((test_rep - nnet.cvar.dimshuffle('x', 0)) ** 2),
                      axis=1, dtype='floatX')

    test_scores = test_dist - nnet.Rvar
    test_stack = T.stack([T.zeros_like(test_scores), test_scores], axis=1)
    test_loss = T.cast(T.sum(T.max(test_stack, axis=1)) / (inputs.shape[0]*nu),
                       dtype='floatX')

    test_y_pred = T.argmax(test_stack, axis=1)
    test_acc = T.cast((T.sum(T.eq(test_y_pred.flatten(), targets),
                             dtype='int32')
                       * 1. / targets.shape[0]), dtype='floatX')

    # Network activation sparsity regularization (with determinisitc=True)
    if Cfg.sparsity_penalty:
        test_sparsity_penalty = ((1 / B) *
                                 get_sparsity_penalty(nnet, inputs,
                                                      Cfg.sparsity,
                                                      mode=Cfg.sparsity_mode,
                                                      deterministic=True))
    else:
        test_sparsity_penalty = T.cast(0, dtype='floatX')

    test_obj = T.cast(floatX(0.5) * l2_penalty + nnet.Rvar + test_loss
                      + test_sparsity_penalty, dtype='floatX')
    nnet.forward = theano.function([inputs, targets],
                                   [test_obj, test_acc, test_scores,
                                    floatX(0.5) * l2_penalty,
                                    test_sparsity_penalty, test_rep,
                                    test_rep_norm, test_loss, nnet.Rvar])


def create_autoencoder(nnet):
    """
    create autoencoder Theano update for network given in argument
    """

    floatX = Cfg.floatX
    B = Cfg.ae_B
    C = Cfg.ae_C
    ndim = nnet.data._X_train.ndim

    if ndim == 2:
        inputs = T.matrix('inputs')
    elif ndim == 4:
        inputs = T.tensor4('inputs')

    final_layer = nnet.all_layers[-1]

    # Backpropagation
    trainable_params = lasagne.layers.get_all_params(final_layer,
                                                     trainable=True)
    prediction = lasagne.layers.get_output(final_layer, inputs=inputs,
                                           deterministic=False)
    # use l2 or binary crossentropy loss (features are scaled to [0,1])
    if Cfg.ae_loss == "l2":
        loss = lasagne.objectives.squared_error(prediction, inputs)
    if Cfg.ae_loss == "ce":
        loss = lasagne.objectives.binary_crossentropy(prediction, inputs)

    scores = T.sum(loss, axis=range(1, ndim), dtype='floatX')
    loss = T.mean(scores)

    # Regularization
    if Cfg.ae_weight_decay:
        l2_penalty = (floatX(0.5) / C) * regularize_network_params(final_layer,
                                                                   l2)
    else:
        l2_penalty = T.cast(0, dtype='floatX')

    # Network activation sparsity regularization
    if Cfg.ae_sparsity_penalty:
        sparsity_penalty = ((1 / B) *
                            get_sparsity_penalty(nnet, inputs, Cfg.ae_sparsity,
                                                 mode=Cfg.ae_sparsity_mode,
                                                 deterministic=False))
    else:
        sparsity_penalty = T.cast(0, dtype='floatX')

    train_obj = loss + l2_penalty + sparsity_penalty
    updates = get_updates(nnet, train_obj, trainable_params,
                          solver=nnet.ae_solver)
    nnet.ae_backprop = theano.function([inputs],
                                       [loss, l2_penalty, sparsity_penalty,
                                        scores],
                                       updates=updates)

    # Forwardpropagation

    test_prediction = lasagne.layers.get_output(final_layer, inputs=inputs,
                                                deterministic=True)
    # use l2 or binary crossentropy loss (features are scaled to [0,1])
    if Cfg.ae_loss == "l2":
        test_loss = lasagne.objectives.squared_error(test_prediction, inputs)
    if Cfg.ae_loss == "ce":
        test_loss = lasagne.objectives.binary_crossentropy(test_prediction,
                                                           inputs)

    test_scores = T.sum(test_loss, axis=range(1, ndim), dtype='floatX')
    test_loss = T.mean(test_scores)

    # Network activation sparsity regularization (with determinisitc=True)
    if Cfg.ae_sparsity_penalty:
        test_sparsity_penalty = ((1 / B) *
                                 get_sparsity_penalty(nnet, inputs,
                                                      Cfg.ae_sparsity,
                                                      mode=Cfg.ae_sparsity_mode,
                                                      deterministic=True))
    else:
        test_sparsity_penalty = T.cast(0, dtype='floatX')

    nnet.ae_forward = theano.function([inputs],
                                      [test_loss, l2_penalty,
                                       test_sparsity_penalty, test_scores,
                                       test_prediction])
