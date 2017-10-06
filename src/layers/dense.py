import lasagne.layers
import theano.tensor as T

from layers import fun
from config import Configuration as Cfg


class DenseLayer(lasagne.layers.DenseLayer):

    # for convenience
    isconv, isbatchnorm, isdropout, ismaxpool, isactivation = (False,) * 5
    issvm = False
    isdense = True

    def __init__(self, incoming_layer, num_units,
                 W=lasagne.init.GlorotUniform(),
                 b=lasagne.init.Constant(0.),
                 name=None, normalize=False, nonlinearity=None, **kwargs):

        lasagne.layers.DenseLayer.__init__(self, incoming_layer, num_units,
                                           name=name, W=W, b=b,
                                           nonlinearity=nonlinearity, **kwargs)
        self.normalize = normalize
        self.inp_ndim = 2
        self.use_dc = True

        if Cfg.compile_lwsvm:
            self.initialize_variables()
            self.compile_methods()

            self.update_fixed_weights()

    def initialize_variables(self):

        fun.initialize_variables_dense(self)

    def compile_methods(self):

        self.norm = fun.compute_norm(self)
        self.get_dual = fun.get_dual(self)
        self.use_average = fun.use_average(self)
        self.separate_weights = fun.separate_weights(self)
        self.initialize_warm_start = fun.initialize_warm_start(self)
        self.allocate_primal_variables = \
            fun.allocate_primal_variables_dense(self)
        self.reinitialize_primal_variables = \
            fun.reinitialize_primal_variables(self)
        self.deallocate_primal = fun.deallocate_primal_variables_dense(self)
        self.get_warm_regularization = fun.warm_regularization(self)

        self.norm_avg = fun.compile_norm_avg(self)
        self.warm_reg_avg = fun.compile_warm_reg_avg(self)

        self.update_fixed_weights = fun.update_fixed_weights(self)

        self.get_input_for = fun.get_input_for(self)

        if Cfg.store_on_gpu:
            self.get_input_batch = fun.input_batch(self)

    def forward_with(self,
                     X,
                     W,
                     b,
                     **kwargs):

        if X.ndim > 2:
            X = X.flatten(2)

        Z = T.dot(X, W) + b.dimshuffle(('x', 0))

        return Z

    def forward_prop(self,
                     X_ccv,
                     X_cvx,
                     **kwargs):

        if X_cvx.ndim > 2:
            X_cvx = X_cvx.flatten(2)
            X_ccv = X_ccv.flatten(2)

        Z_ccv = T.dot(X_cvx, self.W_neg) + T.dot(X_ccv, self.W_pos)
        Z_cvx = T.dot(X_cvx, self.W_pos) + T.dot(X_ccv, self.W_neg) + \
            self.b.dimshuffle(('x', 0))

        return Z_ccv, Z_cvx

    def initialize_primal(self,
                          nnet):
        """ order matters here? to document...
        """

        self.update_fixed_weights()
        self.allocate_primal_variables()
        if Cfg.store_on_gpu:
            fun.precompute_input(self, nnet)
        self.initialize_warm_start()

        # necessary?
        self.reinitialize_primal_variables()

        # # @todo @init
        # for idx in nnet.batch_indices:
        #     _, y = nnet.get_train_batch(idx)
        #     self.initialize_loss(idx, y)

        # need some pre-processing if using DC decomposition
        if self.use_dc:
            fun.prepare_dc_decomposition(nnet, self)
