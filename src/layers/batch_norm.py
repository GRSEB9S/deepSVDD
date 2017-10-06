import lasagne.layers

from config import Configuration as Cfg
from layers import fun


class BatchNorm(lasagne.layers.BatchNormLayer):

    # for convenience
    isdense, isconv, isdropout, ismaxpool, isactivation, issvm = (False,) * 6
    isbatchnorm = True

    def __init__(self,
                 incoming_layer,
                 name=None,
                 **kwargs):

        lasagne.layers.BatchNormLayer.__init__(self, incoming_layer, name=name)

        # self.nnet_inp_ndim = incoming_layer.nnet_inp_ndim
        if Cfg.compile_lwsvm:
            self.initialize_variables()
            self.compile_methods()

    def initialize_variables(self):

        fun.initialize_variables_batchnorm(self)

    def compile_methods(self):

        self.get_input_for = fun.get_input_for(self)
        if self.input_layer.isdense:
            self.separate_gamma = fun.separate_gamma(self)

    def get_output_for(self, input, deterministic=False,
                       batch_norm_use_averages=None,
                       batch_norm_update_averages=None, **kwargs):
        """
        """

        return lasagne.layers.BatchNormLayer\
            .get_output_for(self, input,
                            deterministic=deterministic,
                            batch_norm_use_averages=batch_norm_use_averages,
                            batch_norm_update_averages=
                            batch_norm_update_averages,
                            **kwargs)

    def forward_prop(self,
                     ZIn_ccv,
                     ZIn_cvx,
                     **kwargs):

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(range(ZIn_ccv.ndim - len(self.axes)))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(ZIn_ccv.ndim)]

        beta = self.beta.dimshuffle(pattern)
        gamma_pos = self.gamma_pos.dimshuffle(pattern)
        gamma_neg = self.gamma_neg.dimshuffle(pattern)
        mean = self.mean.dimshuffle(pattern)
        inv_std = self.inv_std.dimshuffle(pattern)

        ZIn_ccv = ZIn_ccv * inv_std
        ZIn_cvx = (ZIn_cvx - mean) * inv_std

        ZOut_cvx = gamma_pos * ZIn_cvx + gamma_neg * ZIn_ccv + beta
        ZOut_ccv = gamma_pos * ZIn_ccv + gamma_neg * ZIn_cvx

        return ZOut_ccv, ZOut_cvx

    def forward_like(self,
                     input,
                     input_fixed,
                     **kwargs):

        out = self.get_output_for(input, deterministic=True)
        out_fixed = self.get_output_for(input_fixed, deterministic=True)

        return out, out_fixed
