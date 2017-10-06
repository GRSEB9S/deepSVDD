import lasagne.layers


class Dimshuffle(lasagne.layers.DimshuffleLayer):

    # for convenience
    isdense, isconv, isdropout, ismaxpool, isactivation, issvm = (False,) * 6
    isbatchnorm = False

    def __init__(self, incoming_layer, name=None, **kwargs):

        lasagne.layers.DimshuffleLayer.__init__(self,
                                                incoming=incoming_layer,
                                                name=name,
                                                **kwargs)
