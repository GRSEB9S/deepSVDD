import lasagne.layers
import theano.tensor as T

from layers import fun


class ReLU(lasagne.layers.NonlinearityLayer):

    # for convenience
    isdense, isbatchnorm, isconv, isdropout, ismaxpool, issvm = (False,) * 6
    isactivation, isrelu = (True,) * 2
    issigmoid = False

    def __init__(self, incoming_layer, name=None,
                 nonlinearity=lasagne.nonlinearities.rectify, **kwargs):

        lasagne.layers.NonlinearityLayer.__init__(self, incoming_layer,
                                                  name=name,
                                                  nonlinearity=nonlinearity,
                                                  **kwargs)

        self.num_units = self.input_shape[1]

        self.compile_methods()

    def compile_methods(self):

        self.get_input_for = fun.get_input_for(self)

    def forward_prop(self, X_ccv, X_cvx, **kwargs):

        # Numerical instabilities when using Z_cvx = T.maximum(X_ccv, X_cvx)
        Z_ccv = X_ccv
        Z_cvx = T.nnet.relu(X_cvx - X_ccv) + Z_ccv

        return Z_ccv, Z_cvx

    def forward_like(self, input, input_fixed, **kwargs):

        return mirror_activations(input, input_fixed)


def mirror_activations(input, input_fixed):

    out_fixed = T.nnet.relu(input_fixed)
    mask = T.grad(cost=None,
                  wrt=input_fixed,
                  known_grads={out_fixed: T.ones_like(out_fixed)})

    out = input * mask

    return out, out_fixed
