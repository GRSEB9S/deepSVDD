import theano.tensor as T

import lasagne.layers
from theano.tensor.signal.pool import pool_2d
from utils.patches import my_pool_2d
from layers import fun

from config import Configuration as Cfg


class MaxPool(lasagne.layers.Pool2DLayer):

    # for convenience
    isdense, isbatchnorm, isconv, isdropout, isactivation, issvm = (False,) * 6
    ismaxpool = True

    def __init__(self, incoming_layer, pool_size, stride=None, pad=(0, 0),
                 ignore_border=True, name=None):

        lasagne.layers.Pool2DLayer.__init__(self, incoming_layer, pool_size,
                                            name=name, mode="max",
                                            stride=stride, pad=pad,
                                            ignore_border=ignore_border)

        self.inp_ndim = 4
        self.initialize_variables()
        self.compile_methods()

    def initialize_variables(self):

        self.inp_ndim = 4
        self.pool_opts = {'ds': self.pool_size,
                          'st': self.stride,
                          'padding': self.pad,
                          'ignore_border': self.ignore_border}

    def compile_methods(self):

        self.get_input_for = fun.get_input_for(self)

    def get_output_for(self, input, **kwargs):

        pooled = my_pool_2d(input, mode='max', **self.pool_opts)
        return pooled

    def forward_prop(self, X_ccv, X_cvx, **kwargs):

        Z_ccv = Cfg.floatX(self.pool_size[0] * self.pool_size[1]) * \
            pool_2d(X_ccv, mode='average_exc_pad', **self.pool_opts)
        Z_cvx = my_pool_2d(X_cvx - X_ccv, mode='max', **self.pool_opts) + Z_ccv

        return Z_ccv, Z_cvx

    def forward_like(self, input, input_fixed, **kwargs):

        return mirror_activations(input, input_fixed, self.pool_size)


def mirror_activations(input, input_fixed, pool_size):

        out_fixed = my_pool_2d(input_fixed, ds=pool_size, ignore_border=True)
        mask = T.grad(cost=None,
                      wrt=input_fixed,
                      known_grads={out_fixed: T.ones_like(out_fixed)})

        masked_input = input * mask
        out = Cfg.floatX(pool_size[0] * pool_size[1]) * \
            pool_2d(masked_input, mode='average_exc_pad', ds=pool_size,
                    ignore_border=True)

        return out, out_fixed
