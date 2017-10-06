import lasagne.layers


class WeightNorm(lasagne.layers.ScaleLayer):

    # for convenience
    isdense, isconv, isdropout, ismaxpool, isactivation, issvm = (False,) * 6
    isbatchnorm = False

    def __init__(self,
                 incoming_layer,
                 scales,
                 shared_axes,
                 name=None,
                 **kwargs):

        lasagne.layers.ScaleLayer.__init__(self,
                                           incoming=incoming_layer,
                                           scales=scales,
                                           shared_axes=shared_axes,
                                           name=name,
                                           **kwargs)
