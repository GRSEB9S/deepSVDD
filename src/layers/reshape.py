import lasagne.layers


class Reshape(lasagne.layers.ReshapeLayer):

    # for convenience
    isdense, isconv, isdropout, ismaxpool, isactivation, issvm = (False,) * 6
    isbatchnorm = False

    def __init__(self, incoming_layer, shape, name=None, **kwargs):

        lasagne.layers.ReshapeLayer.__init__(self,
                                             incoming=incoming_layer,
                                             shape=shape,
                                             name=name,
                                             **kwargs)
