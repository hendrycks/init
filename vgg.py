#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trains and tests a DenseNet on CIFAR-10.

For usage information, call with --help.

Author: Jan Schlüter
"""

import os
from argparse import ArgumentParser
import numpy as np

import theano
import theano.tensor as T
import lasagne
# from lasagne.layers import (InputLayer, Conv2DLayer, FlattenLayer, DenseLayer,
#                             DropoutLayer, Pool2DLayer, GlobalPoolLayer,
#                             NonlinearityLayer, ScaleLayer, BiasLayer)
import lasagne.layers as ll
from lasagne.layers import dnn

# from theano.sandbox.rng_mrg import MRG_RandomStreams
# rng = np.random.RandomState(1)
# theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
# lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))

np.random.seed(100)

def opts_parser():
    usage = "Trains and tests a DenseNet on CIFAR-10."
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '--epochs', type=int, default=150,
        help='Number of training epochs (default: %(default)s)')
    parser.add_argument(
        '--eta', type=float, default=0.00001,
        help='Initial learning rate (default: %(default)s)')
    parser.add_argument(
        '--save-weights', type=str, default=None, metavar='FILE',
        help='If given, save network weights to given .npz file')
    parser.add_argument(
        '--save-errors', type=str, default=None, metavar='FILE',
        help='If given, save train/validation errors to given .npz file')
    parser.add_argument(
        '--resume', action='store_true', default=False,
        help='Should we continue training from some earlier model?')
    parser.add_argument(
        '--init_name', type=str, default="ours",
        help='Weight init name ("ours" is default)')
    parser.add_argument(
        '--nonlinearity_name', type=str, default="relu",
        help='Nonlinearity name (relu default)')
    parser.add_argument(
        '--use_cifar10', action='store_true', default=True,
        help='Use CIFAR-10')
    parser.add_argument(
        '--use_cifar100', action='store_false', dest='use_cifar10',
        help='Use CIFAR-100')
    return parser


def generate_in_background(generator, num_cached=10):
    """
    Runs a generator in a background thread, caching up to `num_cached` items.
    """
    import queue
    queue = queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    # start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        item = queue.get()


class Initializer(object):
    def __call__(self, shape):
        return self.sample(shape)

    def sample(self, shape):
        raise NotImplementedError()

class Ours(Initializer):
    # we choose to use an orthonormal matrix instead of just a random matrix with its columns normalized
    def __init__(self, initializer, gain=1.0):
        self.initializer = initializer
        self.gain = gain

    def sample(self, shape):
        if len(shape) < 2:
            raise RuntimeError("Only shapes of length 2 or more are supported.")

        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return np.asarray(self.gain * q, dtype=theano.config.floatX)

class our_init(Ours):
    def __init__(self, gain=1.0):
        super(our_init, self).__init__(Ours, gain)

# TODO: remove
class GlobalAvgLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(GlobalAvgLayer, self).__init__(incoming, **kwargs)
    def get_output_for(self, input, **kwargs):
        return T.mean(input, axis=(2,3))
    def get_output_shape_for(self, input_shape):
        return input_shape[:2]

def build_vgg(input_var, classes, f=lasagne.nonlinearities.rectify, init_name='ours'):

    if init_name == 'he' or init_name == 'xavier':
        if init_name == 'xavier':
            initializer = lasagne.init.GlorotUniform(gain='relu')
        elif init_name == 'he':
            initializer = lasagne.init.HeNormal(gain='relu')

        # normalizer = dnn.BatchNormDNNLayer

        # network = ll.InputLayer((None, 3, 32, 32), input_var)
        # network = normalizer(ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 64, (3,3), pad=1, W=initializer, nonlinearity=f), p=0.3))
        # network = normalizer(dnn.Conv2DDNNLayer(network, 64, (3,3), pad=1, W=initializer, nonlinearity=f))
        # network = ll.MaxPool2DLayer(network, 2, stride=(2,2))
        # network = normalizer(ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 128, (3,3), pad=1, W=initializer, nonlinearity=f), p=0.4))
        # network = normalizer(dnn.Conv2DDNNLayer(network, 128, (3,3), pad=1, W=initializer, nonlinearity=f))
        # network = ll.MaxPool2DLayer(network, 2, stride=(2,2))
        # network = normalizer(ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 256, (3,3), pad=1, W=initializer, nonlinearity=f), p=0.4))
        # network = normalizer(ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 256, (3,3), pad=1, W=initializer, nonlinearity=f), p=0.4))
        # network = normalizer(dnn.Conv2DDNNLayer(network, 256, (3,3), pad=1, W=initializer, nonlinearity=f))
        # network = ll.MaxPool2DLayer(network, 2, stride=(2,2))
        # network = normalizer(ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, W=initializer, nonlinearity=f), p=0.4))
        # network = normalizer(ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, W=initializer, nonlinearity=f), p=0.4))
        # network = normalizer(dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, W=initializer, nonlinearity=f))
        # network = ll.MaxPool2DLayer(network, 2, stride=(2,2))
        # network = normalizer(ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, W=initializer, nonlinearity=f), p=0.4))
        # network = normalizer(ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, W=initializer, nonlinearity=f), p=0.4))
        # network = normalizer(dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, W=initializer, nonlinearity=f))
        # network = ll.DropoutLayer(ll.FlattenLayer(ll.MaxPool2DLayer(network, 2, stride=(2,2))), p=0.5)
        # network = normalizer(ll.DropoutLayer(ll.DenseLayer(network, num_units=512, W=initializer, nonlinearity=f), p=0.5))
        # network = normalizer(ll.DenseLayer(network, num_units=classes, W=initializer, nonlinearity=lasagne.nonlinearities.softmax))

        network = ll.InputLayer((None, 3, 32, 32), input_var)
        network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 64, (3,3), pad=1, W=initializer, nonlinearity=f), p=0.3)
        network = dnn.Conv2DDNNLayer(network, 64, (3,3), pad=1, W=initializer, nonlinearity=f)
        network = ll.MaxPool2DLayer(network, 2, stride=(2,2))
        network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 128, (3,3), pad=1, W=initializer, nonlinearity=f), p=0.4)
        network = dnn.Conv2DDNNLayer(network, 128, (3,3), pad=1, W=initializer, nonlinearity=f)
        network = ll.MaxPool2DLayer(network, 2, stride=(2,2))
        network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 256, (3,3), pad=1, W=initializer, nonlinearity=f), p=0.4)
        network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 256, (3,3), pad=1, W=initializer, nonlinearity=f), p=0.4)
        network = dnn.Conv2DDNNLayer(network, 256, (3,3), pad=1, W=initializer, nonlinearity=f)
        network = ll.MaxPool2DLayer(network, 2, stride=(2,2))
        network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, W=initializer, nonlinearity=f), p=0.4)
        network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, W=initializer, nonlinearity=f), p=0.4)
        network = dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, W=initializer, nonlinearity=f)
        network = ll.MaxPool2DLayer(network, 2, stride=(2,2))
        network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, W=initializer, nonlinearity=f), p=0.4)
        network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, W=initializer, nonlinearity=f), p=0.4)
        network = dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, W=initializer, nonlinearity=f)
        network = ll.DropoutLayer(ll.FlattenLayer(ll.MaxPool2DLayer(network, 2, stride=(2,2))), p=0.5)
        network = ll.DropoutLayer(ll.DenseLayer(network, num_units=512, W=initializer, nonlinearity=f), p=0.5)
        network = ll.DenseLayer(network, num_units=classes, W=initializer, nonlinearity=lasagne.nonlinearities.softmax)

        # network = ll.InputLayer((None, 3, 32, 32), input_var)
        # network = dnn.Conv2DDNNLayer(network, 64, (3,3), pad=1, W=initializer, nonlinearity=f)
        # network = dnn.Conv2DDNNLayer(network, 64, (3,3), pad=1, W=initializer, nonlinearity=f)
        # network = ll.DropoutLayer(ll.MaxPool2DLayer(network, 2, stride=(2,2)), p=0.4)
        # network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 128, (3,3), pad=1, W=initializer, nonlinearity=f), p=0.4)
        # network = dnn.Conv2DDNNLayer(network, 128, (3,3), pad=1, W=initializer, nonlinearity=f)
        # network = ll.DropoutLayer(ll.MaxPool2DLayer(network, 2, stride=(2,2)), p=0.4)
        # network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 256, (3,3), pad=1, W=initializer, nonlinearity=f), p=0.4)
        # network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 256, (3,3), pad=1, W=initializer, nonlinearity=f), p=0.4)
        # network = dnn.Conv2DDNNLayer(network, 256, (3,3), pad=1, W=initializer, nonlinearity=f)
        # network = ll.DropoutLayer(ll.MaxPool2DLayer(network, 2, stride=(2,2)), p=0.4)
        # network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, W=initializer, nonlinearity=f), p=0.4)
        # network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, W=initializer, nonlinearity=f), p=0.4)
        # network = dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, W=initializer, nonlinearity=f)
        # network = ll.DropoutLayer(ll.MaxPool2DLayer(network, 2, stride=(2,2)), p=0.4)
        # network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, W=initializer, nonlinearity=f), p=0.4)
        # network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, W=initializer, nonlinearity=f), p=0.4)
        # network = dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, W=initializer, nonlinearity=f)
        # network = ll.DropoutLayer(ll.FlattenLayer(ll.MaxPool2DLayer(network, 2, stride=(2,2))), p=0.5)
        # network = ll.DropoutLayer(ll.DenseLayer(network, num_units=512, W=initializer, nonlinearity=f), p=0.5)
        # network = ll.DenseLayer(network, num_units=classes, W=initializer, nonlinearity=lasagne.nonlinearities.softmax)

    elif init_name == 'ours':
        initializer = our_init
        network = ll.InputLayer((None, 3, 32, 32), input_var)
        network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 64, (3,3), pad=1, nonlinearity=f,
                                                         W=initializer(1/np.sqrt(1 + 0*0.7*0.5))), p=0.3)
        network = dnn.Conv2DDNNLayer(network, 64, (3,3), pad=1, nonlinearity=f,
                                         W=initializer(1/np.sqrt(0.5/0.7 + 0*0.5)))
        # we ignore estimating the variance added by the max pool transform
        network = ll.MaxPool2DLayer(network, 2, stride=(2,2))
        network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 128, (3,3), pad=1, nonlinearity=f,
                                                         W=initializer(1/np.sqrt(0.5 + 0*0.6*0.5))), p=0.4)
        network = dnn.Conv2DDNNLayer(network, 128, (3,3), pad=1, nonlinearity=f,
                                         W=initializer(1/np.sqrt(0.5/0.6 + 0*0.5)))
        network = ll.MaxPool2DLayer(network, 2, stride=(2,2))
        network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 256, (3,3), pad=1, nonlinearity=f,
                                                         W=initializer(1/np.sqrt(0.5 + 0*0.6*0.5))), p=0.4)
        network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 256, (3,3), pad=1, nonlinearity=f,
                                                         W=initializer(1/np.sqrt(0.5/0.6 + 0*0.6*0.5))), p=0.4)
        network = dnn.Conv2DDNNLayer(network, 256, (3,3), pad=1, nonlinearity=f,
                                         W=initializer(1/np.sqrt(0.5/0.6 + 0*0.5)))
        network = ll.MaxPool2DLayer(network, 2, stride=(2,2))
        network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, nonlinearity=f,
                                                         W=initializer(1/np.sqrt(0.5 + 0*0.6*0.5))), p=0.4)
        network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, nonlinearity=f,
                                                         W=initializer(1/np.sqrt(0.5/0.6 + 0*0.6*0.5))), p=0.4)
        network = dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, nonlinearity=f,
                                         W=initializer(1/np.sqrt(0.5/0.6 + 0*0.5)))
        network = ll.MaxPool2DLayer(network, 2, stride=(2,2))
        network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, nonlinearity=f,
                                                         W=initializer(1/np.sqrt(0.5 + 0*0.6*0.5))), p=0.4)
        network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, nonlinearity=f,
                                                         W=initializer(1/np.sqrt(0.5/0.6 + 0*0.6*0.5))), p=0.4)
        network = dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, nonlinearity=f,
                                         W=initializer(1/np.sqrt(0.5/0.6 + 0*0.5*0.5)))
        network = ll.DropoutLayer(ll.FlattenLayer(ll.MaxPool2DLayer(network, 2, stride=(2,2))), p=0.5)
        network = ll.DropoutLayer(ll.DenseLayer(network, num_units=512, nonlinearity=f,
                                                    W=initializer(1/np.sqrt(0.5/0.5 + 0*0.5*0.5))), p=0.5)
        network = ll.DenseLayer(network, num_units=classes, nonlinearity=lasagne.nonlinearities.softmax,
                                 W=initializer(1/np.sqrt(0.5/0.5 + 0*0.5)))  # using 0.5 for softmax

        # network = ll.InputLayer((None, 3, 32, 32), input_var)
        # network = dnn.Conv2DDNNLayer(network, 64, (3,3), pad=1, nonlinearity=f,
        #                                              W=initializer(1/np.sqrt(1 + 0.5)))
        # network = dnn.Conv2DDNNLayer(network, 64, (3,3), pad=1, nonlinearity=f,
        #                              W=initializer(1/np.sqrt(0.5 + 0.6*0.5)))
        # # we ignore estimating the variance added by the max pool transform
        # network = ll.DropoutLayer(ll.MaxPool2DLayer(network, 2, stride=(2,2)), p=0.4)
        # network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 128, (3,3), pad=1, nonlinearity=f,
        #                                              W=initializer(1/np.sqrt(0.5/0.6 + 0.6*0.5))), p=0.4)
        # network = dnn.Conv2DDNNLayer(network, 128, (3,3), pad=1, nonlinearity=f,
        #                              W=initializer(1/np.sqrt(0.5/0.6 + 0.6*0.5)))
        # network = ll.DropoutLayer(ll.MaxPool2DLayer(network, 2, stride=(2,2)), p=0.4)
        # network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 256, (3,3), pad=1, nonlinearity=f,
        #                                              W=initializer(1/np.sqrt(0.5/0.6 + 0.6*0.5))), p=0.4)
        # network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 256, (3,3), pad=1, nonlinearity=f,
        #                                              W=initializer(1/np.sqrt(0.5/0.6 + 0.6*0.5))), p=0.4)
        # network = dnn.Conv2DDNNLayer(network, 256, (3,3), pad=1, nonlinearity=f,
        #                              W=initializer(1/np.sqrt(0.5/0.6 + 0.6*0.5)))
        # network = ll.DropoutLayer(ll.MaxPool2DLayer(network, 2, stride=(2,2)), p=0.4)
        # network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, nonlinearity=f,
        #                                              W=initializer(1/np.sqrt(0.5/0.6 + 0.6*0.5))), p=0.4)
        # network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, nonlinearity=f,
        #                                              W=initializer(1/np.sqrt(0.5/0.6 + 0.6*0.5))), p=0.4)
        # network = dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, nonlinearity=f,
        #                              W=initializer(1/np.sqrt(0.5/0.6 + 0.6*0.5)))
        # network = ll.DropoutLayer(ll.MaxPool2DLayer(network, 2, stride=(2,2)), p=0.4)
        # network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, nonlinearity=f,
        #                                              W=initializer(1/np.sqrt(0.5/0.6 + 0.6*0.5))), p=0.4)
        # network = ll.DropoutLayer(dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, nonlinearity=f,
        #                                              W=initializer(1/np.sqrt(0.5/0.6 + 0.6*0.5))), p=0.4)
        # network = dnn.Conv2DDNNLayer(network, 512, (3,3), pad=1, nonlinearity=f,
        #                              W=initializer(1/np.sqrt(0.5/0.6 + 0.5*0.5)))
        # network = ll.DropoutLayer(ll.FlattenLayer(ll.MaxPool2DLayer(network, 2, stride=(2,2))), p=0.5)
        # network = ll.DropoutLayer(ll.DenseLayer(network, num_units=512, nonlinearity=f,
        #                                         W=initializer(1/np.sqrt(0.5/0.5 + 0.5*0.5))), p=0.5)
        # network = ll.DenseLayer(network, num_units=classes, nonlinearity=lasagne.nonlinearities.softmax,
        #                         W=initializer(1/np.sqrt(0.5/0.5 + 0.5)))  # using 0.5 for softmax

    else:
        assert False, "Need 'he', 'xavier', or 'ours' as the init name"

    return network

def train_test(epochs, eta, save_weights, save_errors, resume,
               init_name, nonlinearity_name, use_cifar10, batchsize=128):
    # import (deferred until now to make --help faster)
    import numpy as np
    import theano
    import theano.tensor as T
    import lasagne

    if use_cifar10 is True:
        print('Using CIFAR-10')
        import cifar10 as dataset
        num_classes = 10
    else:
        print('Using CIFAR-100')
        import cifar100 as dataset
        num_classes = 100
    import progress

    # instantiate network
    print("Instantiating network...")
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    if nonlinearity_name == 'relu':
        f = lasagne.nonlinearities.rectify
    elif nonlinearity_name == 'elu':
        f = lasagne.nonlinearities.elu
    elif nonlinearity_name == 'gelu':
        def gelu(x):
            return 0.5 * x * (1 + T.tanh(T.sqrt(2 / np.pi) * (x + 0.044715 * T.pow(x, 3))))
        f = gelu

    network = build_vgg(input_var, num_classes, f, init_name)
    print("%d layers with weights, %d parameters" %
          (sum(hasattr(l, 'W')
               for l in lasagne.layers.get_all_layers(network)),
           lasagne.layers.count_params(network, trainable=True)))

    # load dataset
    print("Loading dataset...")
    X_train, y_train, X_test, y_test = dataset.load_dataset(
        path=os.path.join(os.path.dirname(__file__), 'data'))
    # if validate == 'test':
    X_val, y_val = X_test, y_test
    # elif validate:
    #     X_val, y_val = X_train[-5000:], y_train[-5000:]
    #     X_train, y_train = X_train[:-5000], y_train[:-5000]

    # define training function
    print("Compiling training function...")
    prediction = ll.get_output(network)
    prediction = T.clip(prediction, 1e-7, 1 - 1e-7)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var).mean()
    l2_loss = 5e-4 * lasagne.regularization.regularize_network_params(
        network, lasagne.regularization.l2, {'regularizable': True})
    params = lasagne.layers.get_all_params(network, trainable=True)
    eta = theano.shared(lasagne.utils.floatX(eta), name='eta')
    # updates = lasagne.updates.nesterov_momentum(
    #     loss + l2_loss, params, learning_rate=eta)
    updates = lasagne.updates.adam(
        loss + l2_loss, params, learning_rate=eta)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    l2_fn = theano.function([], l2_loss)

    # define validation/testing function
    print("Compiling testing function...")
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var).mean()
    test_err = 1 - lasagne.objectives.categorical_accuracy(test_prediction,
                                                           target_var).mean()
    test_fn = theano.function([input_var, target_var], [test_loss, test_err])

    start_epoch = 0
    if save_errors:
        errors = []

    if resume is True:
        errors = list(np.load(save_errors)['errors'].reshape(-1))
        for i in range(epochs-1,-1,-1):
            try:
                with np.load(save_weights+'_'+str(i)+'.npz') as f:
                    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
                lasagne.layers.set_all_param_values(network, param_values)
                start_epoch = i+1
                print('Restored!', i, start_epoch)
                break
            except:
                True
        if start_epoch == 0:
            assert False, "could not resume"

    # Finally, launch the training loop.
    print("Starting training...")

    orig_lr = eta.get_value()
    for epoch in range(start_epoch, epochs):
        # eta.set_value(lasagne.utils.floatX(orig_lr * max(0.1 ** (epoch//25), 1e-7)))

        # restoration friendly code
        # drop at half and then at three fourths through training
        if 100 <= epoch < 125:
            eta.set_value(orig_lr * lasagne.utils.floatX(0.1))
        elif epoch >= 125:
            eta.set_value(orig_lr * lasagne.utils.floatX(0.01))

        # In each epoch, we do a full pass over the training data:
        train_loss = 0
        train_batches = len(X_train) // batchsize
        batches = dataset.iterate_minibatches(X_train, y_train, batchsize, shuffle=True)
        # augmentation is mandatory!
        batches = dataset.augment_minibatches(batches)
        batches = generate_in_background(batches)
        batches = progress.progress(
            batches, desc='Epoch %d/%d, Batch ' % (epoch + 1, epochs),
            total=train_batches)
        for inputs, targets in batches:
            train_loss += train_fn(inputs, targets)

        # And possibly a full pass over the validation data:
        # if validate:
        #     val_loss = 0
        #     val_err = 0
        #     val_batches = len(X_val) // batchsize
        #     for inputs, targets in dataset.iterate_minibatches(X_val, y_val, batchsize, shuffle=False):
        #         loss, err = test_fn(inputs, targets)
        #         val_loss += loss
        #         val_err += err
        # else:
        test_loss = 0
        test_err = 0
        test_batches = len(X_test) // batchsize
        for inputs, targets in dataset.iterate_minibatches(X_test, y_test, batchsize, shuffle=False):
            loss, err = test_fn(inputs, targets)
            test_loss += loss
            test_err += err

        # Then we print the results for this epoch:
        train_loss /= train_batches
        l2_loss = l2_fn()
        print("  CE loss:\t%.6f" % train_loss)
        print("  L2 loss:      \t%.6f" % l2_loss)
        print("  Loss:      \t%.6f" % (train_loss+l2_loss))
        if save_errors:
            errors.extend([train_loss, l2_loss])

        # if validate:
        #     val_loss /= val_batches
        #     val_err /= val_batches
        #     print("  validation loss:\t%.6f" % val_loss)
        #     print("  validation error:\t%.2f%%" % (val_err * 100))
        #     if save_errors:
        #         errors.extend([val_loss, val_err])
        # else:
        test_loss /= test_batches
        test_err /= test_batches
        print("  test loss:\t%.6f" % test_loss)
        print("  test error:\t%.2f%%" % (test_err * 100))
        if save_errors:
            errors.extend([test_loss, test_err])

        if epoch % 25 == 0 and epoch > 100:
            # Optionally, we dump the network weights to a file
            if save_weights:
                np.savez(save_weights+'_'+str(epoch), *lasagne.layers.get_all_param_values(network))

            # Optionally, we dump the learning curves to a file
            if save_errors:
                np.savez(save_errors, errors=np.asarray(errors).reshape(epoch+1, -1))

    # After training, we compute and print the test error:
    test_loss = 0
    test_err = 0
    test_batches = len(X_test) // batchsize
    for inputs, targets in dataset.iterate_minibatches(X_test, y_test,
                                                       batchsize,
                                                       shuffle=False):
        loss, err = test_fn(inputs, targets)
        test_loss += loss
        test_err += err
    print("Final results:")
    print("  test loss:\t\t%.6f" % (test_loss / test_batches))
    print("  test error:\t\t%.2f%%" % (test_err / test_batches * 100))

    # we dump the network weights to a file
    np.savez(save_weights, *lasagne.layers.get_all_param_values(network))
    # Optionally, we dump the learning curves to a file
    np.savez(save_errors, errors=np.asarray(errors).reshape(epochs, -1))


def main():
    # parse command line
    parser = opts_parser()
    args = parser.parse_args()

    # run
    train_test(**vars(args))


if __name__ == "__main__":
    main()
