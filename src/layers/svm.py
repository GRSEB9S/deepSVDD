import lasagne.layers
import theano.tensor as T

from layers import fun

from config import Configuration as Cfg


class SVMLayer(lasagne.layers.DenseLayer):

    # for convenience
    isdense, isbatchnorm, isconv, isdropout, isactivation = (False,) * 5
    ismaxpool = False
    issvm = True

    def __init__(self, incoming_layer, num_units,
                 W=lasagne.init.GlorotUniform(),
                 b=lasagne.init.Constant(0.)):

        lasagne.layers.DenseLayer.__init__(self, incoming_layer, num_units,
                                           name="svm", W=W, b=b,
                                           nonlinearity=None)

        self.inp_ndim = 2
        self.n_classes = self.num_units

        if Cfg.compile_lwsvm:
            self.initialize_variables()
            self.compile_methods()

    def initialize_variables(self):

        fun.initialize_variables_svm(self)

    def compile_methods(self):

        self.norm = fun.compute_norm(self)
        self.get_dual = fun.get_dual(self)
        self.use_average = fun.use_average(self)
        self.separate_weights = fun.separate_weights(self)
        self.initialize_warm_start = fun.initialize_warm_start(self)
        self.allocate_primal_variables = \
            fun.allocate_primal_variables_svm(self)
        self.reinitialize_primal_variables = \
            fun.reinitialize_primal_variables(self)
        self.deallocate_primal = fun.deallocate_primal_variables_svm(self)
        self.get_warm_regularization = fun.warm_regularization(self)
        self.norm_avg = fun.compile_norm_avg(self)
        self.warm_reg_avg = fun.compile_warm_reg_avg(self)

        self.get_input_for = fun.get_input_for(self)

        if Cfg.store_on_gpu:
            self.get_input_batch = fun.input_batch(self)

    def forward_with(self, X, W, b, **kwargs):

        if X.ndim > 2:
            X = X.flatten(2)

        Z = T.dot(X, W) + b.dimshuffle(('x', 0))

        return Z

    @staticmethod
    def objective(scores, y_truth):

        t_range = T.arange(y_truth.shape[0])

        y_star, delta = SVMLayer.max_oracle(scores, y_truth)

        # hinge loss summed over samples
        objective = delta + scores[t_range, y_star].sum() - \
            scores[t_range, y_truth].sum()

        # get prediction
        y_pred = T.argmax(scores, axis=1)
        acc = T.sum(T.eq(y_pred, y_truth))

        return objective, acc

    @staticmethod
    def max_oracle(scores,
                   y_truth):

        n_classes = scores.shape[1]
        t_range = T.arange(y_truth.shape[0])

        # classification loss for any combination
        losses = 1. - T.extra_ops.to_one_hot(y_truth, n_classes)

        # get max score for each sample
        y_star = T.argmax(scores + losses, axis=1)

        # compute classification loss for batch
        delta = losses[t_range, y_star].sum()

        return y_star, delta

    def initialize_primal(self,
                          nnet,
                          *args):
        """ initialize primal:
        - warm start: store (W_0, b_0) and initialize (W, b) with warm start
        - allocate variables: allocate W_i, b_i, l_i, l, X_layer, dW and db
        - precompute input: pre-compute input of every batch
        and store it in X_layer attribute
        """

        self.initialize_warm_start()
        self.allocate_primal_variables()

        if Cfg.store_on_gpu:
            fun.precompute_input(self, nnet)

    def forward_prop(self,
                     X_ccv,
                     X_cvx,
                     y,
                     **kwargs):

        y_range = T.arange(y.shape[0])

        # get loss for each class
        scores = self.get_output_for(X_cvx - X_ccv)

        # get standard SVM objective
        objective, _ = self.objective(scores, y)

        # common part in convex and concave part (sum of concave terms)
        common = T.dot(X_cvx, self.W_neg).sum() + \
            T.dot(X_ccv, self.W_pos).sum()

        # convex contribution is max prediction + common part
        # - concave ground truth
        err_cvx = objective + scores[y_range, y].sum() + common \
            - T.sum(self.W_pos[:, y].T * X_cvx) \
            - T.sum(self.W_neg[:, y].T * X_ccv)

        # concave contribution is common part - convex ground truth
        err_ccv = common \
            - T.sum(self.W_neg[:, y].T * X_cvx) \
            - T.sum(self.W_pos[:, y].T * X_ccv)

        return err_ccv, err_cvx
