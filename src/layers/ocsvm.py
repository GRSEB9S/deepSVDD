import lasagne.layers
import theano.tensor as T


class OCSVMLayer(lasagne.layers.DenseLayer):

    # for convenience
    isdense, isbatchnorm, isconv, isdropout, isactivation = (False,) * 5
    ismaxpool = False
    issvm = True

    def __init__(self, incoming_layer,
                 W=lasagne.init.GlorotUniform(),
                 b=lasagne.init.Constant(0.)):

        lasagne.layers.DenseLayer.__init__(self, incoming_layer, num_units=1,
                                           name="ocsvm", W=W, b=b,
                                           nonlinearity=None)

    # methods for SGD training
    @staticmethod
    def objective(scores, y_truth):

        # either zero or positive loss
        stack = T.stack([T.zeros_like(scores), -scores], axis=1)
        losses = T.max(stack, axis=1)

        # loss summed over samples
        objective = losses.sum()

        # get prediction (normal: 0; outlier: 1)
        y_pred = T.argmax(stack, axis=1)
        acc = T.sum(T.eq(y_pred.flatten(), y_truth), dtype='int32')

        return objective, acc
