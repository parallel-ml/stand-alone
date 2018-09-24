"""
    This module provides single Conv2D layer channel wise split.
    Technique used is simple that divide filter into number of
    batches and copy identical inputs onto batch of filters.
"""
from keras.layers import Conv2D, Lambda
from keras.layers.merge import Concatenate


def split(X, num):
    """ Return a list of 3D tensor split by channel. """
    return Lambda(lambda x: [x for _ in range(num)])(X)


def merge(tensors):
    return Concatenate()(tensors)


def conv(tensors, filters, kernal, strides, padding, activation, name):
    size = []
    for _ in range(len(tensors) - 1):
        size.append(filters / len(tensors))
    size.append(filters - filters / len(tensors) * (len(tensors) - 1))

    return [Conv2D(size[i], kernal, strides=strides, padding=padding, activation=activation, name=name + '_conv_' + str(i))(x)
            for i, x in enumerate(tensors)]