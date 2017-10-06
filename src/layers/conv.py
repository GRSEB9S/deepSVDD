import lasagne.layers
import theano.tensor as T

from layers import fun
from config import Configuration as Cfg


class ConvLayer(lasagne.layers.Conv2DLayer):

    # for convenience
    isdense, isbatchnorm, isdropout, ismaxpool, isactivation = (False,) * 5
    issvm = False
    isconv = True

    def __init__(self, incoming_layer, num_filters, filter_size, stride=(1, 1),
                 pad="valid", W=lasagne.init.GlorotUniform(gain='relu'),
                 b=lasagne.init.Constant(0.), flip_filters=True, name=None):

        lasagne.layers.Conv2DLayer.__init__(self, incoming_layer, num_filters,
                                            filter_size, name=name,
                                            stride=stride, pad=pad,
                                            untie_biases=False, W=W, b=b,
                                            nonlinearity=None,
                                            flip_filters=flip_filters,
                                            convolution=T.nnet.conv2d)

        self.inp_ndim = 4
        self.use_dc = False

        if Cfg.compile_lwsvm:
            self.initialize_variables()
            self.compile_methods()

            self.update_fixed_weights()

    def initialize_variables(self):

        fun.initialize_variables_conv(self)

        border_mode = 'half' if self.pad == 'same' else self.pad
        self.conv_opts = {"subsample": self.stride,
                          "border_mode": border_mode,
                          "filter_flip": self.flip_filters}

    def compile_methods(self):

        self.norm = fun.compute_norm(self)
        self.get_dual = fun.get_dual(self)
        self.use_average = fun.use_average(self)
        self.initialize_warm_start = fun.initialize_warm_start(self)
        self.allocate_primal_variables = \
            fun.allocate_primal_variables_conv(self)
        self.reinitialize_primal_variables = \
            fun.reinitialize_primal_variables(self)
        self.deallocate_primal = fun.deallocate_primal_variables_conv(self)
        self.get_warm_regularization = fun.warm_regularization(self)
        self.update_fixed_weights = fun.update_fixed_weights(self)
        self.norm_avg = fun.compile_norm_avg(self)
        self.warm_reg_avg = fun.compile_warm_reg_avg(self)

        self.get_input_for = fun.get_input_for(self)

    def forward_with(self, X, W, b, **kwargs):

        Z = T.nnet.conv2d(X, W, input_shape=None, **self.conv_opts) \
            + b.dimshuffle(('x', 0, 'x', 'x'))

        return Z

    def initialize_primal(self, nnet, *args):

        self.update_fixed_weights()
        self.allocate_primal_variables()
        self.initialize_warm_start()
        self.reinitialize_primal_variables()

        # need some pre-processing if using DC decomposition
        if self.use_dc:
            fun.prepare_dc_decomposition(nnet, self)

    def convolve(self, input, **kwargs):
        border_mode = 'half' if self.pad == 'same' else self.pad
        conved = self.convolution(input, self.W,
                                  None, self.get_W_shape(),
                                  subsample=self.stride,
                                  border_mode=border_mode,
                                  filter_flip=self.flip_filters)
        return conved
