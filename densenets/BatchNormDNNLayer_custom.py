# -*- coding: utf-8 -*-
"""
Preliminary implementation of batch normalization for Lasagne, using cuDNN.

Author: Jan Schlüter
Based on: https://gist.github.com/marcociccone/1f19a197df97871288535c4ed40927a0
See: https://github.com/Lasagne/Lasagne/issues/743
"""

from lasagne.layers import BatchNormLayer
from lasagne import init
import theano
from theano.sandbox.cuda import dnn

class BatchNormDNNLayer(BatchNormLayer):
    def __init__(self, incoming, mode='spatial', epsilon=1e-4, alpha=0.1,
                 beta=init.Constant(0), gamma=init.Constant(1),
                 mean=init.Constant(0), inv_std=init.Constant(1), **kwargs):
        assert mode in ('per-activation', 'spatial'), \
            "Mode parameter should be 'per-activation' or 'spatial'"

        if mode == 'per-activation':
            axes = (0,)
        elif mode == 'spatial':
            axes = (0,) + tuple(range(2, len(getattr(incoming, 'output_shape', incoming))))
        super(BatchNormDNNLayer, self).__init__(incoming, axes, epsilon, alpha, beta, gamma, mean, inv_std, **kwargs)
        self.mode = mode

    def get_output_for(self, input, deterministic=False,
                       batch_norm_use_averages=None,
                       batch_norm_update_averages=None, **kwargs):
        # Decide whether to use the stored averages or mini-batch statistics
        if batch_norm_use_averages is None:
            batch_norm_use_averages = deterministic
        use_averages = batch_norm_use_averages

        # Decide whether to update the stored averages
        if batch_norm_update_averages is None:
            batch_norm_update_averages = not deterministic
        update_averages = batch_norm_update_averages

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(range(input.ndim - len(self.axes)))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]
        # and prepare the converse pattern removing those broadcastable axes
        unpattern = [d for d in range(input.ndim) if d not in self.axes]

        # call cuDNN if needed, obtaining normalized outputs and statistics
        if not use_averages or update_averages:
            # cuDNN requires beta/gamma tensors; create them if needed
            shape = tuple(s for (d, s) in enumerate(input.shape)
                          if d not in self.axes)
            gamma = self.gamma or theano.tensor.ones(shape)
            beta = self.beta or theano.tensor.zeros(shape)
            (normalized,
             input_mean,
             input_inv_std) = dnn.dnn_batch_normalization_train(
                    input, gamma.dimshuffle(pattern), beta.dimshuffle(pattern),
                    self.mode, self.epsilon)

        # normalize with stored averages, if needed
        if use_averages:
            mean = self.mean.dimshuffle(pattern)
            inv_std = self.inv_std.dimshuffle(pattern)
            gamma = 1 if self.gamma is None else self.gamma.dimshuffle(pattern)
            beta = 0 if self.beta is None else self.beta.dimshuffle(pattern)
            normalized = (input - mean) * (gamma * inv_std) + beta

        # update stored averages, if needed
        if update_averages:
            # Trick: To update the stored statistics, we create memory-aliased
            # clones of the stored statistics:
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_inv_std = theano.clone(self.inv_std, share_inputs=False)
            # set a default update for them:
            # running_mean.default_update = ((1 - self.alpha) * running_mean +
            #                                self.alpha * input_mean.dimshuffle(unpattern))
            running_inv_std.default_update = ((1 - self.alpha) *
                                              running_inv_std +
                                              self.alpha * input_inv_std.dimshuffle(unpattern))
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            dummy = 0 * (running_mean + running_inv_std).dimshuffle(pattern)
            normalized = normalized + dummy

        return normalized
