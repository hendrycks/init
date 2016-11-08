#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trains and tests a DenseNet on CIFAR-10.

For usage information, call with --help.

Author: Jan Schlüter
"""

import sys
import os
from argparse import ArgumentParser


def opts_parser():
    usage = "Trains and tests a DenseNet on CIFAR-10."
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '-L', '--depth', type=int, default=40,
        help='Network depth in layers (default: %(default)s)')
    parser.add_argument(
        '-k', '--growth-rate', type=int, default=12,
        help='Growth rate in dense blocks (default: %(default)s)')
    parser.add_argument(
        '--dropout', type=float, default=0,
        help='Dropout rate (default: %(default)s)')
    parser.add_argument(
        '--augment', action='store_true', default=True,
        help='Perform data augmentation (enabled by default)')
    parser.add_argument(
        '--no-augment', action='store_false', dest='augment',
        help='Disable data augmentation')
    parser.add_argument(
        '--validate', action='store_true', default=False,
        help='Perform validation on validation set (disabled by default)')
    parser.add_argument(
        '--no-validate', action='store_false', dest='validate',
        help='Disable validation')
    parser.add_argument(
        '--validate-test', action='store_const', dest='validate',
        const='test', help='Perform validation on test set')
    parser.add_argument(
        '--epochs', type=int, default=300,
        help='Number of training epochs (default: %(default)s)')
    parser.add_argument(
        '--eta', type=float, default=0.1,
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


def train_test(depth, growth_rate, dropout, augment, validate, epochs,
               eta, save_weights, save_errors, resume, nonlinearity_name,
               use_cifar10, batchsize=64):
    # import (deferred until now to make --help faster)
    import numpy as np
    import theano
    import theano.tensor as T
    import lasagne

    import densenet_fast_custom as densenet  # or "import densenet" for slower version
    if use_cifar10 is True:
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
    network = densenet.build_densenet(input_var=input_var, depth=depth, classes=num_classes,
                                      growth_rate=growth_rate, dropout=dropout,
                                      nonlinearity_name=nonlinearity_name)
    print("%d layers with weights, %d parameters" %
          (sum(hasattr(l, 'W')
               for l in lasagne.layers.get_all_layers(network)),
           lasagne.layers.count_params(network, trainable=True)))

    # load dataset
    print("Loading dataset...")
    X_train, y_train, X_test, y_test = dataset.load_dataset(
        path=os.path.join(os.path.dirname(__file__), 'data'))
    if validate == 'test':
        X_val, y_val = X_test, y_test
    elif validate:
        X_val, y_val = X_train[-5000:], y_train[-5000:]
        X_train, y_train = X_train[:-5000], y_train[:-5000]

    # define training function
    print("Compiling training function...")
    prediction = lasagne.layers.get_output(network)
    prediction = T.clip(prediction, 1e-7, 1 - 1e-7)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # note: The paper says 1e-4 decay, but 1e-4 in Torch is 5e-5 elsewhere
    l2_loss = 1e-4 * lasagne.regularization.regularize_network_params(
            network, lasagne.regularization.l2, {'trainable': True})
    params = lasagne.layers.get_all_params(network, trainable=True)
    eta = theano.shared(lasagne.utils.floatX(eta), name='eta')
    updates = lasagne.updates.nesterov_momentum(
            loss + l2_loss, params, learning_rate=eta, momentum=0.9)
    # updates = lasagne.updates.adam(
    #         loss + l2_loss, params, learning_rate=eta)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # define validation/testing function
    print("Compiling testing function...")
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    update_var_prediction = lasagne.layers.get_output(network, deterministic=True, batch_norm_update_averages=True,
                                                      batch_norm_use_averages=False)
    loss_var_update = lasagne.objectives.categorical_crossentropy(update_var_prediction, target_var)
    loss_var_update = loss_var_update.mean()
    update_var_fn = theano.function([input_var, target_var], loss_var_update)
    test_loss = test_loss.mean()
    test_acc = lasagne.objectives.categorical_accuracy(test_prediction,
                                                       target_var).mean()
    test_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    l2_fn = theano.function([], l2_loss)

    with np.load("./wider_07_100.npz") as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    # Finally, launch the training loop.
    print("Starting training...")
    if save_errors:
        errors = []

    val_err = 0
    val_acc = 0
    val_batches = len(X_test) // batchsize
    for inputs, targets in dataset.iterate_minibatches(X_test, y_test,
                                                       batchsize,
                                                       shuffle=False):
        err, acc = test_fn(inputs, targets)
        val_err += err
        val_acc += acc
    if validate or True:  # HACK: validate on test set, for debugging
        print("  validation loss:\t%.6f" % (val_err / val_batches))
        print("  validation error:\t%.2f%%" % (
            100 - val_acc / val_batches * 100))

    for epoch in range(5):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = len(X_train) // batchsize
        batches = dataset.iterate_minibatches(X_train, y_train, batchsize,
                                              shuffle=True)
        if augment:
            batches = dataset.augment_minibatches(batches)
            batches = generate_in_background(batches)
        batches = progress.progress(
                batches, desc='Epoch %d/%d, Batch ' % (epoch + 1, epochs),
                total=train_batches)
        for inputs, targets in batches:
            train_err += update_var_fn(inputs, targets)

        # And possibly a full pass over the validation data:
        if validate:
            val_err = 0
            val_acc = 0
            val_batches = len(X_val) // batchsize
            for inputs, targets in dataset.iterate_minibatches(X_val, y_val,
                                                               batchsize,
                                                               shuffle=False):
                err, acc = test_fn(inputs, targets)
                val_err += err
                val_acc += acc
        else:
            # HACK: validate on test set, for debugging
            val_err = 0
            val_acc = 0
            val_batches = len(X_test) // batchsize
            for inputs, targets in dataset.iterate_minibatches(X_test, y_test,
                                                               batchsize,
                                                               shuffle=False):
                err, acc = test_fn(inputs, targets)
                val_err += err
                val_acc += acc

        # Then we print the results for this epoch:
        print("  training loss:\t%.6f" % (train_err / train_batches))
        l2_err = l2_fn()
        print("  L2 loss:      \t%.6f" % l2_err)
        if save_errors:
            errors.extend([train_err / train_batches, l2_err])
        if validate or True:  # HACK: validate on test set, for debugging
            print("  validation loss:\t%.6f" % (val_err / val_batches))
            print("  validation error:\t%.2f%%" % (
                100 - val_acc / val_batches * 100))
            if save_errors:
                errors.extend([val_err / val_batches,
                               100 - val_acc / val_batches * 100])

        if save_weights and epoch % 20 == 0:
            np.savez(save_weights, *lasagne.layers.get_all_param_values(network))
            print('Saved')

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = len(X_test) // batchsize
    for inputs, targets in dataset.iterate_minibatches(X_test, y_test,
                                                       batchsize,
                                                       shuffle=False):
        err, acc = test_fn(inputs, targets)
        test_err += err
        test_acc += acc
    print("Final results:")
    print("  test loss:\t\t%.6f" % (test_err / test_batches))
    print("  test error:\t\t%.2f%%" % (
        100 - test_acc / test_batches * 100))


def main():
    # parse command line
    parser = opts_parser()
    args = parser.parse_args()

    # run    
    train_test(**vars(args))


if __name__ == "__main__":
    main()
